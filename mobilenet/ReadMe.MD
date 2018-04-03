How to run the program and retrain the network

1) Install TensorFlow
https://www.tensorflow.org/install/

2) Add images to the folder fruits like this:
~..\dataset\FolderName\ImageName.jpg

3) Set IMAGE_SIZE and ARCHITECTURE
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

4) Run this script (Add --how_many_training_steps=500 \ if you want to limit the training steps)
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/"${ARCHITECTURE}" \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=..\dataset

5) Run this script for a selected image in the test folder
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/test_img/ImageName.jpg