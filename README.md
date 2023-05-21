# MoviePosterClassifyer
AI FINAL PROJECT
Took me a while to figure out what I wanted to do, but once I did I found a Kaggle CSV that looked like what I wanted. But I realized the links to the posters was depreciated so I made a python script with beautifulsoup4 to scrape the web to replace the links with ones to IMDB.
Once I did that I set up a script to download all of the posters into corresponding genre folders.
Running this NN with 12 genres, my identification accuracy peaked at around 27%. This was most likely due to each poster being in multiple genre folders (so even if it identifies a movie as action, if it’s looking at the adventure version it would be wrong). So now I have to implement multi-label classification, which might take some time.
I just spent a while trying to implement a depreciated interface that didn’t handle images. Oops! Now my data is currently in a folder containing subfolders that have the genre name and images of the posters, but I need to write me a python file to convert it into a structured format.
Did it! But it’s out of order with the file structure, so I need to support it. It took a lot of Excel finagling, but it’s ready to be processed now.
I just spent a while figuring out the flow_from_dataframe method to create image generators, but now I finally have my network training.
I am now writing a file that loads in the neural network and takes file names as parameters and outputs predictions.
Initial model took too long to train:
model = models.Sequential([
    data_aug,
    layers.experimental.preprocessing.Rescaling(1./255), #does what the flatten method would do
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(1028, activation='tanh'),
    layers.Dense(12, activation='sigmoid')
])
New one below
model = models.Sequential([
    data_aug,
    layers.experimental.preprocessing.Rescaling(1./255), #does what the flatten method would do
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(512, activation='tanh'),
    layers.Dense(12, activation='sigmoid')
])

Currently my Network evaluates at around 33% for the validation. Seeing as it has to identify it as multiple out of 12 genres, that seems pretty good. I think there were some issues in my dataset. 1. The ratio of genres was off, leading to a lot of drama and action movies. For example, there were five times as many Drama’s as there were sci-fi. 2. There were also video game banners in the dataset I downloaded, which probably threw off the network when it comes to classifying animated movies. If I did this again, I think I would put more effort into tailoring my data to fit with my goal. But learning Multi-label classification was really enjoyable, despite being WAY harder then I expected. That had to do with the fact that to read in my data, I had to use a depreciated module of Keras. It also wasn’t letting me load my model if I saved it as an h5.
To run the classifier, just run it with the names of .jpg files as parameters. I ran my classifier on some other films, and this is what it spit out: For Coco, it was Action, Adventure, Animation, Comedy. For Parasite, Action, Drama. (this makes since because the poster is one that seems kinda ambiguous. For Se7en, I got Action, Drama, and Horror. Indiana Jones and The Last Crusade was Action, Comedy, and Drama. Finally, It saw Everything Everywhere All At Once as a Action, Adventure, and Comedy.  I’m impressed with the networks ability to classify comedy movies. Anyway, that is my project. I’ve had a lot of fun plugging in photos of my life and seeing what type of movie it thinks it is.

    
