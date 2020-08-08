import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import VGGEncoder, Decoder
from style_swap import style_swap

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main():
    parser = argparse.ArgumentParser(description='Style Swap by Pytorch')
    parser.add_argument('--content_dir', '-c', type=str,
                        default='/data/jsy/code/MAST/data/default_data/content',
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style_dir', '-s', type=str,
                        default='/data/jsy/code/MAST/data/default_data/style',
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_dir', '-o', type=str, default='res/combine_test',
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--patch_size', '-p', type=int, default=3,
                        help='Size of extracted patches from style features')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # set model
    e = VGGEncoder().to(device)
    d = Decoder()
    d.load_state_dict(torch.load(args.model_state_path))
    d = d.to(device)

    c_name_list = os.listdir(args.content_dir)
    c_name_list.sort()
    s_name_list = os.listdir(args.style_dir)
    s_name_list.sort()
    for c_name in c_name_list:
        for s_name in s_name_list:
            c_path = os.path.join(args.content_dir, c_name)
            s_path = os.path.join(args.style_dir, s_name)
            print(f'processing [{c_path}] and [{s_path}]')
            c = Image.open(c_path)
            s = Image.open(s_path)
            c_tensor = trans(c).unsqueeze(0).to(device)
            s_tensor = trans(s).unsqueeze(0).to(device)
            with torch.no_grad():
                cf = e(c_tensor)
                sf = e(s_tensor)
                style_swap_res = style_swap(cf, sf, args.patch_size, 1)
                out = d(style_swap_res)

            out_denorm = denorm(out, device)

            c_basename = os.path.splitext(c_name)[0]
            s_basename = os.path.splitext(s_name)[0]
            out_path = os.path.join(args.output_dir, f'{c_basename}_{s_basename}.bmp')
            save_image(out_denorm, out_path, nrow=1, padding=0)
            print(f'[{out_path}] saved...')


if __name__ == '__main__':
    main()
