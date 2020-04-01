import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import numpy as np
import os
from tqdm import tqdm
import argparse

from PIL import Image
import torch
import cv2
import h5py
import sys
sys.path.append('.')
sys.path.append('./inspection_scripts')
from modelsdir.AFAR import AFAR
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from gradcam import CamExtractor
from guided_backprop import GuidedBackprop


def generate_cam(model, input_image, target_pspi):
    """function from Guided_Gradcam"""
    extractor = CamExtractor(model, 0)
    # Full forward pass
    # conv_output is the output of convolutions at specified layer
    # model_output is the final output of the model (1, 1000)
    conv_output, model_output = extractor.forward_pass(input_image)
    # Target for backprop
    target_pspi = torch.tensor(target_pspi, dtype=torch.float32)[None,None].cuda()
    # Zero grads
    model.zero_grad()
    # Backward pass with specified target
    model_output.backward(gradient=target_pspi, retain_graph=True)
    # Get hooked gradients
    guided_gradients = extractor.gradients.data.cpu().numpy()[0]
    # Get convolution outputs
    target = conv_output.data.cpu().numpy()[0]
    # Get weights from gradients
    weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
    # Create empty numpy array for cam
    cam = np.ones(target.shape[1:], dtype=np.float32)
    # Multiply each weight with its conv output and then, sum
    for i, w in enumerate(weights):
        cam += w * target[i, :, :]
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                   input_image.shape[3]), Image.ANTIALIAS))/255
    return cam


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask
    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

class GuidedGradCam:
    def __init__(self, path, epoch):
        # Get params
        self.afar = AFAR.AFAR(num_outputs=1, pretrain=False)
        self.afar.load(path, epoch)
        self.pretrained_model = self.afar.model
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = True

    def guided_grad_cam(self, numpy_img, target_pspi, file_name_to_export='test', save=False):
        # prep image for the network
        prep_img = torch.from_numpy(cv2.resize(numpy_img, (200, 200))[None]/255).float().unsqueeze_(0)
        prep_img = prep_img.requires_grad_().cuda()
        
        # Grad cam
        # Generate cam mask
        cam = generate_cam(self.pretrained_model, prep_img, target_pspi)

        # Guided backprop
        GBP = GuidedBackprop(self.pretrained_model)
        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_pspi)

        # Guided Grad cam
        cam_gb = guided_grad_cam(cam, guided_grads)
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        if save:
            save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
            save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
        return cam_gb, grayscale_cam_gb

if __name__ == '__main__':
    # https://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
    # https://stackoverflow.com/a/47166787
    np.random.seed(100)
    parser = argparse.ArgumentParser(
        description='Hovercraft')
    parser.add_argument('-regina_images', type=str, default='/project/6005917/zhaosh/pain/artifacts/frames/frames_regina_FAN_optflow_warp_corner_pad_stdface.hdf5')
    parser.add_argument('-unbc_images', type=str, default='/project/6005917/zhaosh/pain/artifacts/frames/frames_unbc_images_2019-09-08-21-40-4519147925.hdf5')
    parser.add_argument('-results_file', type=str, default='/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/afar_training_results.hdf5')
    parser.add_argument('-ggc_file', type=str, default='/home/zhaosh/scratch/GGC.h5')
    parser.add_argument('-display_dataset', nargs='+', default=['na', 'dementia', 'healthy'])
    args = parser.parse_args()
    model_paths = ['/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/0_2019-09-15-19-56-36',
            '/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/1_2019-09-15-19-56-34/',
            '/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/2_2019-09-15-19-56-49/',
            '/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/3_2019-09-15-19-56-43/',
            '/project/6005917/zhaosh/pain/artifacts/training_artifacts/training_logs/train_afar_regina.py/regina_FAN_warp_corner_none_1_rmsprop_1e-4_1e-3_0.5_plateau_True_1_mean0_ut_rht_rdt_uv_rhv_rdv_19363864/4_2019-09-15-19-56-50/']
    model_epochs = [47, 32, 16, 46, 10]
    with h5py.File(args.regina_images, 'r') as regina_h5,\
            h5py.File(args.unbc_images, 'r') as unbc_h5,\
            h5py.File(args.results_file, 'r') as results_h5:
            
            #h5py.File(args.ggc_file, 'r') as ggc_h5:

        # load data from h5
        x = results_h5['gt_pspi'][()]
        y = results_h5['pred_pspi'][()]

        dataset = results_h5['dataset'][()].astype(str)
        condition = results_h5['condition'][()].astype(str)
        subject = results_h5['subject'][()].astype(str)
        task = results_h5['task'][()].astype(str)
        frame_num = results_h5['frame_num'][()]
        fold_num = results_h5['fold_num'][()]
        
        regina_hq_video_subjects = [9, 11, 12, 13, 17, 18, 19, 21, 22, 23, 24, 28, 34, 38, 39, 41, 43, 44, 45, 46, 55, 60,
                                    62, 63, 65, 67, 69, 71, 73, 74, 75, 79, 85, 86, 90, 92, 94, 102]
        #ggc = (ggc_h5['gray_ggc'][...].squeeze()*255.0).astype(np.uint8)

        # subsample image indices
        sample_idx = [0]
        for i in range(1, x.shape[0]):
            last_i = sample_idx[-1]
            if (condition[i] in args.display_dataset) and \
                                      (dataset[i] != 'regina' or int(subject[i]) in regina_hq_video_subjects) and \
                                      (i - last_i > 10 or
                                      x[i] != x[last_i] or
                                      x[i] > 3 or
                                      dataset[i] != dataset[last_i] or
                                      subject[i] != subject[last_i] or
                                      task[i] != task[last_i]):
                sample_idx.append(i)
        x = x[sample_idx]
        y = y[sample_idx]
        dataset = dataset[sample_idx]
        condition = condition[sample_idx]
        subject = subject[sample_idx]
        task = task[sample_idx]
        frame_num = frame_num[sample_idx]
        fold_num = fold_num[sample_idx]

        # add random noise so points spread out
        rand_x = x + np.random.uniform(-0.4, 0.4, x.shape)

        # name for display
        names = ['{}-{}-{}{}{}'.format(dset, cond, sub, t, f_num)
                 for dset, cond, sub, t, f_num in
                 tqdm(zip(dataset, condition, subject, task, frame_num))]

        dset_cmap = {'healthy': 2, 'dementia': 9, 'na': 5}
        c = np.array(list(map(lambda con: dset_cmap.get(con), condition)))

        norm = plt.Normalize(1, 10)
        cmap = plt.cm.jet

        fig, ax = plt.subplots(figsize=(12, 8))
        sc = plt.scatter(rand_x, y, c=c, s=2, cmap=cmap, norm=norm)

        annot = ax.annotate("", xy=(0, 0), xytext=(-60, -15), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=3"))
        annot.set_visible(False)

        def update_annot(ind):
            """
            show annotation text and image for index of a given point
            :param ind:
            :return:
            """
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}-gt[{:.1f}]pred[{:.1f}]".format(names[ind["ind"][0]], x[ind["ind"][0]], y[ind["ind"][0]])
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.8)

            # find annotation image given dataset, task, subject
            dset = unbc_h5 if dataset[ind["ind"][0]] == 'unbc' else regina_h5
            grp = dset[list(filter(lambda d: '{:03d}'.format(int(subject[ind["ind"][0]])) in d[:5].lower() and
                task[ind["ind"][0]] in d.lower(), list(dset.keys())))[0]]
            arr_img = grp['frames'][grp['frame_num'][()] == frame_num[ind["ind"][0]], :, :, :][0]
            GGC = GuidedGradCam(model_paths[fold_num[ind["ind"][0]]], model_epochs[fold_num[ind["ind"][0]]])
            # get guided grad cam image
            _, ggc_img = GGC.guided_grad_cam(cv2.cvtColor(arr_img, cv2.COLOR_BGR2GRAY), 1, file_name_to_export='{}_{}_{}_{}'.format(dataset[ind["ind"][0]], subject[ind["ind"][0]], task[ind["ind"][0]], frame_num[ind["ind"][0]]), save=False)
            #ggc_img = cv2.resize(ggc[ind['ind'][0]], (160, 160))
            ggc_img = cv2.resize((ggc_img.squeeze()*255.0).astype(np.uint8), (160, 160))
            ggc_img = cv2.applyColorMap(ggc_img, cv2.COLORMAP_JET)
            ggc_img[:,:,0] = 0
            ggc_img = cv2.addWeighted(arr_img, 0.7, cv2.cvtColor(ggc_img, cv2.COLOR_BGR2RGB), 0.3, 0)
            imagebox = OffsetImage(np.concatenate([arr_img, ggc_img], axis=1), zoom=1.0)
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, pos,
                                xybox=(55., 90.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.1,
                                arrowprops=dict(
                                    arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                )
            ax.artists = []
            ax.add_artist(ab)

        def hover(event):
            """hover event call back"""
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    # display
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        ax.artists = []
                        fig.canvas.draw_idle()

        def click(event):
            """click event call back"""
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                save_image_dir = 'saved_image'
                if cont:
                    # save image
                    if not os.path.isdir(save_image_dir):
                        os.makedirs(save_image_dir)
                    img = Image.fromarray(ax.artists[0].offsetbox.get_data())
                    save_path = os.path.join(save_image_dir, annot.get_text()+'.png')
                    img.save(save_path)
                    print("Saved at {}".format(save_path))

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_release_event", click)
        plt.show()
    
    
