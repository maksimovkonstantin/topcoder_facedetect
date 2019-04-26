import shutil
import os

def main():
    path = '/wdata'
    pretrained = 'pretrained'
    link_to_pretrained = ' https://download.pytorch.org/models/resnet50-19c8e357.pth'
    
    mxnet_log = 'https://www.dropbox.com/s/3ql6rbjr14pf2mk/log?dl=1'
    mxnet_model = 'https://www.dropbox.com/s/5p32ipi65069z79/model-0000.params?dl=1'
    mxnet_symbol = 'https://www.dropbox.com/s/yb3cuo780phyomo/model-symbol.json?dl=1'

    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    pretrained_path = os.path.join(path, pretrained)
    os.mkdir(pretrained_path)

    load_commnad = 'wget {} -P {}'.format(link_to_pretrained, pretrained_path)
    os.system(load_commnad)
    print('Resnet pretrained loaded')
    
    tmp_save_path = os.path.join(pretrained_path, mxnet_log.split('/')[-1].split('?')[0])
    load_commnad = 'wget {} -O {}'.format(mxnet_log, tmp_save_path)
    print(load_commnad)
    os.system(load_commnad)
    tmp_save_path = os.path.join(pretrained_path, mxnet_model.split('/')[-1].split('?')[0])
    load_commnad = 'wget {} -O {}'.format(mxnet_model, tmp_save_path)
    os.system(load_commnad)
    
    tmp_save_path = os.path.join(pretrained_path, mxnet_symbol.split('/')[-1].split('?')[0])
    load_commnad = 'wget {} -O {}'.format(mxnet_symbol, tmp_save_path)
    os.system(load_commnad)
    
    print('Insigth face loaded')


if __name__ == '__main__':
    main()