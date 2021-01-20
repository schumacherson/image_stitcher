# Image Stitcher

This is an aplication capable of performing the stitching algorithm to multiple images.

The basic idea is to realize the following steps:

- Read all the images;
- Get their keypoints and features;
- Match the keypoints between two images;
- Generate a mosaic of images, stitching the images with a number of matches beyond a threshold

You can execute this code via terminal, by going to the directory where the stitching.py folder is located, and using the command line ```python stitching.py```.

There are four arguments of which you can use:

- ```--images``` (REQUIRED) The directory where your images are stored. Example:  ```python stitching.py --images /home/user/documents/images```
- ```--scale``` Scales the resolution of the images. The run time of the algorithm is based in the resolution and quantity of images. The scale factor should be 0 < s < 1. (Default is 1);
- ```--ext``` The extension of the images. (Default is .jpg) 
- ```--threshold``` The minimum number of matches between two images to consider them related. Images with bigger resolution will have more keypoints, so two images will have more matches. If the algorithm is not stitching the images properly, lower this number. (Default is 50) 

The final result, which is an image with the same extension of the source images, will be created in the directory used in the ```--images``` argument.

Esta é uma aplicaço capaz de realizar o algoritimo de stitching entre múltiplas imagens.

A ideia básica é realizar os seguintes passos:

- Ler todas as imagens;
- Extrair seus 'keypoints' e 'features'
- Realizar a correspondência entre os keypoints das imagens duas a duas;
- Gerar um mosaico de imagens, 'costurando' as imagens com um número de correspondências maior que um valor estipulado

Você pode executar esse código pelo terminal, indo até o diretório em que se encontra o arquivo 'stitching.py' e usando a linha de comando ```python stitching.py```.

- ```--images``` (OBRIGATÓRIO) É informado o diretório em que as imagens de entrada estão localizadas. Exemplo:  ```python stitching.py --images /home/user/documents/images```
- ```--scale``` Aplica um fator de escala na resoluço imagens. O tempo de execução do algoritmo depende da resolução e da quantidade de imagens. O fator de escala deve ser 0 < s < 1. (O padrão é 1);
- ```--ext``` A extensão das imagens. (O padrão é .jpg) 
- ```--threshold``` O número mínimo de correspondências entre duas imagens para que elas sejam consideradas relacionadas. Imagens com resoluçes maiores terão mais 'keypoints', portanto duas imagens terão mais correspondências. Se o algoritmo no está 'costurando' as imagens, abaixe esse número. (O padrão é 50)

Este o meu primeiro algoritmo compartilhado pelo github, portanto não segui nenhum padrão, e o código pode ter estruturas duvidosas, porém tentei ser o mais didático possível na produção deste código. 
