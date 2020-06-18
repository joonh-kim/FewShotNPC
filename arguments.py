import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Few-shot image classification')
    parser.add_argument('--path', default='/home/jk/Documents/FastAdaptationCC')

    parser.add_argument('--data_set', default='miniimagenet', help='miniimagenet, CUB')
    parser.add_argument('--miniimagenet_data_path', default='/work/miniImagenet', help='SDD/HDD path')
    parser.add_argument('--backbone', default='Conv128', help='Conv64/128, ResNet12')
    parser.add_argument('--classifier', default='Ours', help='Ours, Cosine, ArcFace')
    parser.add_argument('--scale_factor', type=int, default=30)
    parser.add_argument('--margin', type=int, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--base_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=3200, help='3200 for Conv128, 1600 for Conv64, 384 for ResNet12')
    parser.add_argument('--num_class', type=int, default=64, help='100 for CUB, 64 for miniimagenet')

    parser.add_argument('--split', default='novel', help='novel, val')
    parser.add_argument('--iteration', type=int, default=600, help='iteration number for few-shot tasks')

    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_support', type=int, default=1)
    parser.add_argument('--n_query', type=int, default=15)

    parser.add_argument('--adaptation_step', type=int, default=50)
    parser.add_argument('--adaptation_scale_factor', type=int, default=30)
    parser.add_argument('--adaptation_lr', type=float, default=0.0005)

    return parser.parse_args()
