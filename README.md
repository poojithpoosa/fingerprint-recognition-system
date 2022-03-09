# fingerprint-recognition-system

Fingerprint recognition is an automated procedure for determining an individual's identification based on a comparison of stored fingerprint data to input. It is one of the most used biometrics for computer system authentication. They are imprints or patterns on the human finger. These imprints become more noticeable as one ages, yet the structures remain constant. Numerous factors contribute to the popularity of fingerprint recognition technologies. One significant benefit of using them as it is widely used by the legal community. It is the most cost-effective, time-efficient, reliable, and easy method of identifying a person. Due to the rarity of two persons having same fingerprints, fingerprint identification is universally acknowledged as a very accurate technique of authentication. 

In this project, a novel method of using four preprocessing techniques—histogram equalization, Gabor filter, thinning, and cropping—is used to improve the accuracy of the machine learning algorithms.

The feature extraction is done by using the deep learning model known as VGG19 and the classification is done by four ML algorithms: random forest, SVM, KNN, and nave bayes.

The accuracy of the algorithms was improved significantly. All the algorithms gave an accuracy score above 90% on the test set, and random forest and SVM were the highest, scoring 99% and 98%, respectively. 

# dataset:
“The Hong Kong Polytechnic University Contactless 2D to Contact-based 2D Fingerprint Images Database”. 

Link: http://www4.comp.polyu.edu.hk/~csajaykr/fingerprint.htm

first folder consists of samples from the original database.


# packages :

Os

Tensorflow

keras

opencv

skimage

PIL

matplotlib

tqdm

sklearn
