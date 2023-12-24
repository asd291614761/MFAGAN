import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr_G', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_D', type=float, default=0.0001, help='learning rate')
parser.add_argument('--train_batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--test_batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--size', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

parser.add_argument('--load', type=str, default='./res/convnextv2_base_1k_224_ema.pt')
parser.add_argument('--train_data_root', type=str, default='../data_new/VT5000/Train', help='the training datasets root')
parser.add_argument('--val_data_root', type=str, default='../data_new/VT5000/Test', help='the value datasets root')
parser.add_argument('--test_data_root', type=str, default='../data_new/', help='the test datasets root')
parser.add_argument('--maps_path', type=str, default='./maps/', help='the test datasets root')

parser.add_argument('--result_save_path', type=str, default='./res/', help='the path to save models and logs')

parser.add_argument('--model_G_Parameter', type=str, default='./Parameter/Best_Parameter.pth', help='load model path')
parser.add_argument('--Parameter_save_path', type=str, default='./Parameter/', help='saved model path')

opt = parser.parse_args()
