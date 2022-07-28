# import the opencv library
from termios import VDISCARD
import cv2
import os
import argparse
from test import test
import torchvision.io as io
import torchvision.transforms as transforms
from model import Net
import torch

def read_image(filepath):
    image = io.read_image(filepath, io.ImageReadMode.RGB)
    return image

def resize_bicubic(src, h, w):
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image

def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = resize_bicubic(src, h, w)
    return image

def write_image(filepath, src):
    io.write_png(src, filepath)

def rgb2ycbcr(src):
    R = src[0]
    G = src[1]
    B = src[2]
    
    ycbcr = torch.zeros(size=src.shape)
    # *Intel IPP
    # ycbcr[0] = 0.257 * R + 0.504 * G + 0.098 * B + 16
    # ycbcr[1] = -0.148 * R - 0.291 * G + 0.439 * B + 128
    # ycbcr[2] = 0.439 * R - 0.368 * G - 0.071 * B + 128
    # *Intel IPP specific for the JPEG codec
    ycbcr[0] =  0.299 * R + 0.587 * G + 0.114 * B
    ycbcr[1] =  -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    ycbcr[2] =  0.5 * R - 0.41869 * G - 0.08131 * B + 128
    
    # Y in range [16, 235]
    ycbcr[0] = torch.clip(ycbcr[0], 16, 235)
    # Cb, Cr in range [16, 240]
    ycbcr[[1, 2]] = torch.clip(ycbcr[[1, 2]], 16, 240)
    ycbcr = ycbcr.type(torch.uint8)
    return ycbcr


def ycbcr2rgb(src):
    Y = src[0]
    Cb = src[1]
    Cr = src[2]

    rgb = torch.zeros(size=src.shape)
    # *Intel IPP
    # rgb[0] = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    # rgb[1] = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    # rgb[2] = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    # *Intel IPP specific for the JPEG codec
    rgb[0] = Y + 1.402 * Cr - 179.456
    rgb[1] = Y - 0.34414 * Cb - 0.71414 * Cr + 135.45984
    rgb[2] = Y + 1.772 * Cb - 226.816

    rgb = torch.clip(rgb, 0, 255)
    rgb = rgb.type(torch.uint8)
    return rgb

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def view(scale, img_file, version, d):
    pad = 6

    device = torch.device(d)



    MODEL_PATH=f"models/ver {version}/trained_model_x{scale}.pth"
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        lr_image = read_image(img_file)
        bicubic_image = upscale(lr_image, scale)
        bicubic_image = bicubic_image[:, pad:-pad, pad:-pad]
        #write_image(f"{output_folder}/bicubic/frame_{count}_x{scale}.jpg", bicubic_image)

        bicubic_image = upscale(lr_image, scale)
        bicubic_image = rgb2ycbcr(bicubic_image)  #should be rgb2ycbcr
        bicubic_image = norm01(bicubic_image)
        bicubic_image = torch.unsqueeze(bicubic_image, dim=0)
        bicubic_image = bicubic_image.to(device)
        #print(bicubic_image.size())

        sr_image = torch.zeros(size=bicubic_image.shape)
        sr_image = sr_image[:, :, pad:-pad, pad:-pad]
        #print(bicubic_image.shape, model(bicubic_image[:,0,:,:]).shape, sr_image.shape, sr_image[:,0,:,:].shape)
        sr_image[:,0,:,:] = model(bicubic_image[:,0,:,:])
        sr_image[:,1,:,:] = bicubic_image[:,1,pad:-pad,pad:-pad]
        sr_image[:,2,:,:] = bicubic_image[:,2,pad:-pad,pad:-pad]
        #print("here")

    sr_image = denorm01(sr_image)

    sr_image = sr_image.type(torch.uint8)
    sr_image = sr_image[0]
    sr_image = ycbcr2rgb(sr_image)

    write_image(f"testdata/Video SR/temp/sr.png", sr_image)

def save_sr_video(test_video, scale, version, device):
    # define a video capture object
    vid = cv2.VideoCapture(test_video)
    count=0
    while(vid.isOpened()):
        count=count+1
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        print(f"processing frame {count}...")
        cv2.imwrite(f"testdata/Video SR/temp/frame{count}_lr.jpg", frame)
        view(scale, f"testdata/temp/frame{count}.jpg", version, d)
        if count==1:
            frame = cv2.imread(f"testdata/Video SR/temp/sr.png")
            height, width, layers = frame.shape  
            video = cv2.VideoWriter(f"{test_video}_x{scale}_sr.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height)) 
        video.write(cv2.imread(cv2.imread(f"testdata/Video SR/temp/sr.png")))
        os.remove(f"testdata/Frames/lr/frame{count}.jpg")
        os.remove(f"testdata/Video SR/temp/sr.png")
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument('--vid_file', type=str, required=True)
parser.add_argument('--scale', type=int, required=True)
parser.add_argument('--ver', type=int, required=True)
parser.add_argument('--device', type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

save_sr_video(args.vid_file, args.scale, args.ver, args.device)