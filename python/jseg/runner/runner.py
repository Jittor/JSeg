import time
import jittor as jt
from tqdm import tqdm
import jseg
import datetime
from jseg.config import get_cfg, save_cfg
from jseg.utils.registry import build_from_cfg, MODELS, SCHEDULERS, DATASETS, HOOKS, OPTIMS
from jseg.utils.general import build_file, current_time, sync, check_file, check_interval, parse_losses, search_ckpt, np2tmp
from jseg.utils.visualize import visualize_result


class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.max_epoch = cfg.max_epoch
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None) ^ (
            self.max_epoch is None), "You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
        self.efficient_val = cfg.efficient_val

        self.model = build_from_cfg(cfg.model, MODELS)
        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,
                                    MODELS,
                                    named_params=self.model.named_parameters(),
                                    model=self.model)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer, OPTIMS, params=params)
        self.scheduler = build_from_cfg(cfg.scheduler,
                                        SCHEDULERS,
                                        optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,
                                            DATASETS,
                                            drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val, DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test, DATASETS)

        self.logger = build_from_cfg(self.cfg.logger,
                                     HOOKS,
                                     work_dir=self.work_dir)

        save_file = build_file(self.work_dir, prefix="config.yaml")
        save_cfg(save_file)

        self.iter = 0
        self.epoch = 0

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        if jt.rank == 0:
            self.logger.print_log(self.model)

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)

        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()

    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch >= self.max_epoch
        else:
            return self.iter >= self.max_iter

    def run(self):
        self.logger.print_log("Start running")

        while not self.finish:
            self.train()

    def test_time(self):
        warmup = 10
        rerun = 100
        self.model.train()
        for batch_idx, (data) in enumerate(self.train_dataset):
            break
        images = data['img']
        img_metas = data['img_metas']
        gt = data['gt_semantic_seg']
        print("warmup...")
        for i in tqdm(range(warmup)):
            losses = self.model(img=images,
                                img_metas=img_metas,
                                gt_semantic_seg=gt)
            all_loss, losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter, self.epoch, by_epoch=True)
        jt.sync_all(True)
        print("testing...")
        start_time = time.time()
        for i in tqdm(range(rerun)):
            losses = self.model(img=images,
                                img_metas=img_metas,
                                gt_semantic_seg=gt)
            all_loss, losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter, self.epoch, by_epoch=True)
        jt.sync_all(True)
        batch_size = len(gt) * jt.world_size
        ptime = time.time() - start_time
        fps = batch_size * rerun / ptime
        print("FPS:", fps)

    def train(self):
        self.model.train()

        start_time = time.time()
        for batch_idx, (data) in enumerate(self.train_dataset):
            images = data['img']
            img_metas = data['img_metas']
            gt = data['gt_semantic_seg']
            losses = self.model(img=images,
                                img_metas=img_metas,
                                gt_semantic_seg=gt)
            all_loss, losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter, self.epoch, by_epoch=False)

            self.iter += 1

            if check_interval(self.iter, self.log_interval):
                batch_size = len(gt) * jt.world_size
                ptime = time.time() - start_time
                fps = batch_size * (batch_idx + 1) / ptime
                eta_time = (self.total_iter - self.iter) * ptime / (batch_idx +
                                                                    1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(name=self.cfg.name,
                            lr=self.optimizer.cur_lr(),
                            iter=self.iter,
                            epoch=self.epoch,
                            batch_idx=batch_idx,
                            batch_size=batch_size,
                            total_loss=all_loss,
                            fps=fps,
                            eta=eta_str)
                data.update(losses)
                data = sync(data)
                # is_main use jt.rank==0, so it's scope must have no jt.Vars
                if jt.rank == 0:
                    self.logger.log(data)

            if check_interval(self.iter, self.checkpoint_interval):
                self.save()
            if check_interval(self.iter, self.eval_interval):
                self.val()
                self.model.train()
            if self.finish:
                break

        self.epoch += 1

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
        else:
            self.logger.print_log("Validating....")
            if self.model.is_training():
                self.model.eval()
            results = []
            for _, (data) in tqdm(enumerate(self.val_dataset)):
                images = data['img']
                img_metas = data['img_metas']
                result = self.model(images, img_metas, return_loss=False)
                jt.sync_all(True)
                if isinstance(result, list):
                    if self.efficient_val:
                        result = [np2tmp(_) for _ in result]
                    results.extend(result)
                else:
                    if self.efficient_val:
                        result = np2tmp(result)
                    results.append(result)

            self.val_dataset.evaluate(results,
                                      metric='mIoU',
                                      logger=self.logger)

    @jt.no_grad()
    @jt.single_process_scope()
    def test(self, save_dir):
        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            self.model.eval()
            for _, (data) in tqdm(enumerate(self.test_dataset)):
                images = data['img']
                img_metas = data['img_metas']
                results = self.model(images, img_metas, return_loss=False)
                visualize_result(results[0],
                                 palette=self.test_dataset.PALETTE,
                                 save_dir=save_dir,
                                 file_name=img_metas[0]['ori_filename'])

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta": {
                "jseg_version": jseg.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "save_time": current_time(),
                "config": self.cfg.dump()
            },
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }

        save_file = build_file(self.work_dir,
                               prefix=f"checkpoints/ckpt_{self.iter}.pkl")
        jt.save(save_data, save_file)
        print("saved")

    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta", dict())
            self.epoch = meta.get("epoch", self.epoch)
            self.iter = meta.get("iter", self.iter)
            self.max_iter = meta.get("max_iter", self.max_iter)
            self.max_epoch = meta.get("max_epoch", self.max_epoch)
            self.scheduler.load_parameters(resume_data.get(
                "scheduler", dict()))
            self.optimizer.load_parameters(resume_data.get(
                "optimizer", dict()))
        if ("model" in resume_data):
            state_dict = resume_data["model"]
        elif ("state_dict" in resume_data):
            state_dict = resume_data["state_dict"]
        else:
            state_dict = resume_data
        self.model.load_parameters(state_dict)
        self.logger.print_log(
            f"Missing key: {self.model.state_dict().keys() - state_dict.keys()}"
        )
        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)
