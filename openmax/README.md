# Arguments
## Openmax
  |        name       |type |       available values        |                     help                      |
  |:-----------------:|:---:|:-----------------------------:|:----------------------------------------------|
  |       gpu-id      | str |range(0,the max number of gpus)|               gpu number to use               |
  |     batch-size    | int |               -               |      batch size for training the model        |
  |       model       | str |         res18,resnext         |           architecture for training           |
  |     train-data    | str |           cifar40             |       the name of in-distribution data        |
  |      in-data      | str |           cifar40             |       the name of in-distribution data        |
  |     pos-label     | int |             {0, 1}            |               positive label                  |
  |     data-root     | str |               -               |    directory where the all data is located    |
  |     save-path     | str |               -               |   directory where the results will be saved   |
  |     model-path    | str |               -               | directory where the trained model is located  |
  |  train-class-num  | int |    		    -                   |             number of train class             |
  |  weibull-tail     | int |               -               |                data used in testing           |
  |  weibull-alpha    | int |               -               |        Classes used in testing                |
  |  weibull-threshold|float|               -               |           threshold used in testing           |
  |     distance      | str |     euclidean, eucos          |                  distance type                |
  |     eu-weight     |float|               -               |                euclidean ratio                |



# RUN
## Validation
```
# Examples
python validation.py --model res18 --in-data cifar40 --batch-size 128 --pos-label 0 --train-class-num 40 --distance eucos --eu-weight 5e-3 --save-path ./save-path/ --data-root ./data-root/ --model-path ./model-path/ --gpu-id 0
```

## Test
```
# Examples
python test.py --model res18 --in-data cifar40 --batch-size 128 --pos-label 0 --train-class-num 40 --weibull-tail 20 --weibull-alpha 40 --distance eucos --eu-weight 5e-3 --save-path ./save-path/ --data-root ./data-root/ --model-path ./model-path/ --gpu-id 0 
```
