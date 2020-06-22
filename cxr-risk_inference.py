#--------------------------------------------------------------
# CXR-Risk inference example
# Michael Lu
# 
# Please read the README.md at https://github.com/michaeltlu/cxr-risk for important information about this code.
#
# Manuscript: Lu MT, Ivanov A, Mayrhofer T, Hosny A, Aerts HJWL, Hoffmann U. Deep learning to assess long-term mortality from chest radiographs. JAMA Network Open. 2019;2(7):e197416. doi:10.1001/jamanetworkopen.2019.7416
#	      https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2738349
# 
#--------------------------------------------------------------

# uncomment to force CPU only inference (no GPU)
# import os
# os.environ['CUDA_VISIBLE_DEVICES']=''

# fastai installation instructions at https://github.com/fastai/fastai   
# this code tested with fastai 1.0.55
from fastai.vision import *

# cadene pretrained model repository 0.7.4 at https://github.com/Cadene/pretrained-models.pytorch 
import pretrainedmodels 

# requires python 3.6+
import pathlib

# ------------------paths and filenames--------------------------------
path = pathlib.Path.cwd()
development_folder = path / 'development'							# development folder contains development images, labels, and model
model_fn = 'cxr-risk_v1'									# model filename without .pth, located in development/models 
test_folder = path / 'test_images'								# test image folder
output_fn = path / 'output' / 'output.csv'							# filename for output
development_fn = development_folder / 'dummy_dataset'/ 'dummy_dataset.csv'			# development dataset placeholder
valid_fn = development_folder / 'dummy_dataset' / 'dummy_valid.csv'				# list of validation dataset images
bs = 2												# batch size of 2 requires ~1.5 GB of GPU RAM or ~3 GB of CPU RAM

#-------------dummy development and testing datasets------------------
df = pd.read_csv(development_fn)
# need to replace ImageList with ImageItemList for fastai versions <1.046
src = (ImageList.from_csv(development_folder, development_fn, folder='', suffix='') 
       .split_by_fname_file(valid_fn)
       .label_from_df()) 
tfms = get_transforms(do_flip=False, flip_vert=False, max_lighting=0.3, max_zoom=1.2, max_warp=0., max_rotate=2.5)
data = (src.transform(tfms, size=224)
            .add_test_folder(test_folder=test_folder)
            .databunch(bs=bs, no_check=True).normalize(imagenet_stats))

#--------------------CNN and inference--------------------------------
# cadene inceptionv4 https://github.com/Cadene/pretrained-models.pytorch/tree/master/pretrainedmodels/models
# modified from hwasiti https://forums.fast.ai/t/lesson-5-advanced-discussion/30865/40?u=hwasiti
def get_model(pretrained=True, model_name = 'inceptionv4', **kwargs ): 
    if pretrained:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    else:
        arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    return arch

def get_cadene_model(pretrained=True, **kwargs ): 
    return fastai_inceptionv4

custom_head = create_head(nf=2048*2, nc=37, ps=0.75, bn_final=False) 
fastai_inceptionv4 = nn.Sequential(*list(children(get_model(model_name = 'inceptionv4'))[:-2]),custom_head) 

# create cnn, load pretrained model, and put in evaluation mode
# replace cnn_learner with create_cnn for fastai versions <1.0.47
learn = cnn_learner(data, get_cadene_model, metrics=accuracy)	
learn.load(model_fn)
learn.model.eval()

# inference with test time augmentation (TTA) 
preds_tta,y_tta,losses_tta = learn.TTA(scale=1.05, ds_type=DatasetType.Test, with_loss=True)

#----------------write probabilities to csv---------------------------
def output_preds_csv(item_array, preds_array, destination_csv:Path, overwrite:bool=False):
    if len(item_array) != len(preds_array):
        print(f'item_array and preds_array are different lengths. No csv written')
        return
    if overwrite is True:
        mode = 'w'
    else:
        mode = 'x'
    output_file = open(destination_csv, mode) # mode 'x' will fail if file exists    
    for i, (a, b) in enumerate(zip(item_array, preds_array)):
        output_file.write(('{0}: {1}, {2} \n'.format(i, a, b)))
    output_file.close()
    print('csv with', len(item_array), 'lines written to', destination_csv)    
    return

# output filenames and results to csv. results are presented as [x, y] -- y is the risk probability.
items_test = data.test_ds.items
preds_test_tta = preds_tta.numpy()
output_preds_csv(item_array=items_test, preds_array=preds_test_tta, destination_csv=output_fn, overwrite=True)
