# Use this to test the model. 
import torch
import argparse
from metrics import calc_psnr
from model import Net
from datasets import EvalDataset

def test(scale, version):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testdata_file = f"testdata/Image SR/set5_x{scale}.h5"
    test_dataset=EvalDataset(testdata_file)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                            shuffle=True)

    MODEL_PATH=f"models/ver {version}/trained_model_x{scale}.pth"
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    sum_psnr = 0
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            #print(lr.size(),hr.size(),sr.size())
            sum_psnr += calc_psnr(hr, sr, max_val=1)

    print("scale", scale, "PSNR", sum_psnr.cpu().numpy() / len(test_loader))

parser = argparse.ArgumentParser()
parser.add_argument('--ver', type=int, required=True)
args = parser.parse_args()

test(2, args.ver)
test(3, args.ver)
test(4, args.ver)