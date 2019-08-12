# ISIC Skin Lesion Analysis Towards Melanoma Detection Competition
Excerpt from ISIC (source: https://challenge2019.isic-archive.com/)

Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Dermoscopy is a skin imaging modality that has demonstrated improvement for diagnosis of skin cancer compared to unaided visual inspection. However, clinicians should receive adequate training for those improvements to be realized. In order to make expertise more widely available, the International Skin Imaging Collaboration (ISIC) has developed the ISIC Archive, an international repository of dermoscopic images, for both the purposes of clinical training, and for supporting technical research toward automated algorithmic analysis by hosting the ISIC Challenges.

## Getting Started

The dataset given was extremely raw and based on other models that I've created I knew that I first needed to separate the images based on the ISIC_2019_Training_GroundTruth.csv. 
One needs to separate the images by Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), Dermatofibroma, Vascular lesion, Squamous cell carcinoma, None of the others; however, the ground truth did not specify the differences in Melanoma so my classification was binary (subject to be updated based-9)

### Prerequisites

What things you need to install in order to test code

```
pip3 install tensorflow tensorflow-gpu Keras cv2 numpy 
```

### Installing

A step by step series of examples that tell you how to get a development env running

1) Change the name of the path to match your path

```
DATADIR = "/home/YOU/YourImages"
```

2) Run dlMel.py

```
if __name__ == '__main__': 
    print('run it')
    #or however you want to run the code, jupyter,etc.
```
3) Should get something like this: 
![ISIC_model_as_of_10082019](https://raw.githubusercontent.com/byhay1/ML-ISIC_2019-Skin_Lesion_Analysis_Towards_Melanoma_Detection/master/screenshots-ISIC/Screenshot%20from%202019-08-11%2023-55-29.png)

## Running the tests

If you want to test an outside image, use the following: 
```
def preparation(fpath): 
    IMG_SIZE = 100
    img_ary = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    new_ary = cv2.resize(img_ary, (IMG_SIZE, IMG_SIZE)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
#your saved model
model = tf.keras.models.load_model("MODEL-NAME.model")

prediction = model.predict([preparation(IMAGE.jpg)])
print(CATEGORIES[int(prediction[0][0])])
```

## Built With

* [TensorFlow](https://www.tensorflow.org/) - Machine Learning Platform
* [Keras](https://keras.io/) - Deep Learning Library/API
* [Python3.7](https://www.python.org/) - Language Used


## Authors

* **Byron Hayes** - *Initial work* - [BeeSting](https://github.com/byhay1)
* **Thitti Sirinopwongsagon** 
* **Nathan Lock**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Youtuber: sentdex
* Stackoverflow, and literally all of the code from others I looked through to develop my fileseparator program (coming soon!) for data science

