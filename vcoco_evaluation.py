import sys
sys.path.append('/users/PCS0256/lijing/spatially-conditioned-graphs/vcoco/v-coco')
from vsrl_eval import VCOCOeval
from cache_template import CacheTemplate
vsrl_annot_file = '/users/PCS0256/lijing/spatially-conditioned-graphs/vcoco/v-coco/data/vcoco/vcoco_test.json'
coco_file = '/users/PCS0256/lijing/spatially-conditioned-graphs/vcoco/v-coco/data/instances_vcoco_all_2014.json'
split_file = '/users/PCS0256/lijing/spatially-conditioned-graphs/vcoco/v-coco/data/splits/vcoco_test.ids'
det_file = '/users/PCS0256/lijing/spatially-conditioned-graphs/vcoco_cache/vcoco_results.pkl'
vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
vcocoeval._do_eval(det_file, ovr_thresh=0.5)