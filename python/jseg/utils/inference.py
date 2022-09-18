import cv2
import os
import jittor as jt
from jseg.runner import Runner
from jseg.config import init_cfg, update_cfg, get_cfg
from jseg.datasets.pipelines import Compose


class InferenceSegmentor:
    def __init__(self, config_file, checkpoint_file, save_dir):
        init_cfg(config_file)
        if len(checkpoint_file) > 0:
            update_cfg(resume_path=checkpoint_file)

        self.runner = Runner()
        self.runner.model.eval()
        self.transforms = Compose(get_cfg().test_pipeline[1:])
        self.palette = self.runner.val_dataset.PALETTE
        self.runner.model.CLASSES = self.runner.val_dataset.CLASSES
        self.runner.model.PALETTE = self.runner.val_dataset.PALETTE
        self.save_dir = save_dir

    def load_img(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = cv2.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    @jt.no_grad()
    @jt.single_process_scope()
    def infer(self, img):
        data = dict(img=img)
        data = self.transforms(self.load_img(data))
        data['img'][0] = data['img'][0].unsqueeze(0)
        results = self.runner.model(**data, return_loss=False, rescale=True)
        results = self.runner.model.show_result(img, results, out_file=os.path.join(self.save_dir, img[img.rfind('/') + 1:]))
        return results
