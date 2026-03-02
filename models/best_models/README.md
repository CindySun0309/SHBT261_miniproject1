# Best Models

## svm
- run_key  : svm_img64_C10_augFalse
- accuracy : 67.36%
- top5_acc : 84.18%
- macro_f1 : 0.531
- config   : img=64, epochs=N/A, lr=N/A, opt=rbf, aug=False

## resnet18
- run_key  : resnet18_img224_ep5_lr0.001_adamw_augFalse
- accuracy : 97.0%
- top5_acc : 99.77%
- macro_f1 : 0.954
- config   : img=224, epochs=5, lr=0.001, opt=adamw, aug=False

## efficientnet_b0
- run_key  : efficientnet_b0_img224_ep10_lr0.001_adam_augTrue
- accuracy : 97.39%
- top5_acc : 99.85%
- macro_f1 : 0.962
- config   : img=224, epochs=10, lr=0.001, opt=adam, aug=True

## vit_small
- run_key  : vit_small_img224_ep10_lr0.001_adamw_augFalse
- accuracy : 98.23%
- top5_acc : 99.69%
- macro_f1 : 0.974
- config   : img=224, epochs=10, lr=0.001, opt=adamw, aug=False

