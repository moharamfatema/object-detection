# 1. Open Images Dataset

- [1. Open Images Dataset](#1-open-images-dataset)
  - [1.1. Overview of Open Images V6](#11-overview-of-open-images-v6)
  - [1.2. Complexity and Diversity](#12-complexity-and-diversity)
  - [1.3. Data organization](#13-data-organization)
    - [1.3.1. Image-level labels](#131-image-level-labels)
    - [1.3.2. Bounding boxes](#132-bounding-boxes)
    - [1.3.3. Object segmentations](#133-object-segmentations)
    - [1.3.4. Visual relationships](#134-visual-relationships)
    - [1.3.5. Localized narratives](#135-localized-narratives)
    - [1.3.6. Point-level labels](#136-point-level-labels)
  - [1.4. Class definitions](#14-class-definitions)
  - [1.5. Preparing the dataset](#15-preparing-the-dataset)
- [Results](#results)
  - [Faster R-CNN ResNet-50](#faster-r-cnn-resnet-50)
    - [Using confidence threshold = 0.5](#using-confidence-threshold--05)
    - [Using confidence threshold = 0.75](#using-confidence-threshold--075)
    - [Using confidence threshold = 0.9](#using-confidence-threshold--09)
  - [Faster R-CNN MobileNet-V3](#faster-r-cnn-mobilenet-v3)
    - [No threshold](#no-threshold)
    - [Using confidence threshold = 0.5](#using-confidence-threshold--05-1)
    - [Using confidence threshold = 0.75](#using-confidence-threshold--075-1)
    - [Using confidence threshold = 0.9](#using-confidence-threshold--09-1)
  - [SSD](#ssd)
    - [Using confidence threshold = 0.1](#using-confidence-threshold--01)
    - [Using confidence threshold = 0.5](#using-confidence-threshold--05-2)
    - [Using confidence threshold = 0.75](#using-confidence-threshold--075-2)
    - [Using confidence threshold = 0.9](#using-confidence-threshold--09-2)

## 1.1. Overview of Open Images V6

Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives:

- It contains a total of 16M bounding boxes for 600 object classes on 1.9M images, making it the largest existing dataset with object location annotations. The boxes have been largely manually drawn by professional annotators to ensure accuracy and consistency. The images are very diverse and often contain complex scenes with several objects (8.3 per image on average).

- Open Images also offers visual relationship annotations, indicating pairs of objects in particular relations (e.g. "woman playing guitar", "beer on table"), object properties (e.g. "table is wooden"), and human actions (e.g. "woman is jumping"). In total it has 3.3M annotations from 1,466 distinct relationship triplets.

- In V5 segmentation masks for 2.8M object instances in 350 classes were added. Segmentation masks mark the outline of objects, which characterizes their spatial extent to a much higher level of detail.

- In V6 675k localized narratives were added: multimodal descriptions of images consisting of synchronized voice, text, and mouse traces over the objects being described. (Note we originally launched localized narratives only on train in V6, but since July 2020 we also have validation and test covered.)

- In v7 66.4M point-level labels over 1.4M images were added, covering 5,827 classes. These labels provide sparse pixel-level localization and are suitable for zero/few-shot semantic segmentation training and evaluation.
Finally, the dataset is annotated with 61.4M image-level labels spanning 20,638 classes.

- Finally, the dataset is annotated with 59.9M image-level labels spanning 19,957 classes.

The creators believe that having a single dataset with unified annotations for image classification, object detection, visual relationship detection, instance segmentation, and multimodal image descriptions will enable to study these tasks jointly and stimulate progress towards genuine scene understanding.

```json
{
    "name": "open-images-v6",
    "zoo_dataset": "fiftyone.zoo.datasets.base.OpenImagesV6Dataset",
    "dataset_type": "fiftyone.types.dataset_types.OpenImagesV6Dataset",
    "num_samples": 20527,
    "downloaded_splits": {
        "validation": {
            "split": "validation",
            "num_samples": 20527
        }
    },
    "classes": [
        "Accordion",
        "Adhesive tape",
        "Aircraft",
        "Airplane",
        "Alarm clock",
        "Alpaca",
        "Ambulance",
        "Animal",
    //    ...
        "Wrench",
        "Zebra",
        "Zucchini"
    ]
}
```

## 1.2. Complexity and Diversity

![plots](fig1.jpeg)

The plots above show the distributions of object centers in normalized image coordinates for various sets of Open Images and other related datasets. The Open Images Train set, which contains most of the data, and Challenge sets show a rich and diverse distribution of a complexity in a similar ballpark to the COCO dataset. This is also confirmed when considering the number of objects per image and their area distribution (plots below). While we improved the density of annotation in the smaller validation and test sets from V4 to V5, their center distribution is simpler and closer to PASCAL 2012. We recommend users to report results on the Challenge set, which offers the hardest performance test for object detectors. We thank Ross Girshick for suggesting this type of visualizations and for correcting the figure in their LVIS paper, which displayed a plot for the validation set without knowing that it was not representative of the whole dataset, and included an intensity scaling artifact that exaggerated its peakiness.

![plots](fig2.jpeg)

## 1.3. Data organization

The dataset is split into a training set (9,011,219 images), a validation set (41,620 images), and a test set (125,436 images). The images are annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives as described below.

### 1.3.1. Image-level labels

Table 1 shows an overview of the image-level labels in all splits of the dataset. All images have machine generated image-level labels automatically generated by a computer vision model similar to Google Cloud Vision API. These automatically generated labels have a substantial false positive rate.

![table](fig3.png)

Moreover, the validation and test sets, as well as part of the training set have human-verified image-level labels. Most verifications were done with in-house annotators at Google. A smaller part was done by crowd-sourcing from Image Labeler: Crowdsource app, g.co/imagelabeler. This verification process practically eliminates false positives (but not false negatives: some labels might be missing from an image). The resulting labels are largely correct and we recommend to use these for training computer vision models. Multiple computer vision models were used to generate the samples (not just the one used for the machine-generated labels) which is why the vocabulary is significantly expanded (#classes column in Table 1).

As a result of our annotation process, each image is annotated both with verified positive image-level labels, indicating some object classes are present, and with verified negative image-level labels, indicating some classes are absent. All other classes which are not explicitly marked as positive or negative for an image are not annotated. The verified negative labels are reliable and can be used during training and evaluation of image classifiers.

Overall, there are 20,638 distinct classes with image-level labels. Trainable classes are those with at least 100 positive human-verifications in the V7 training set. Based on this definition, 9,668 classes are considered trainable and machine-generated labels cover 9,068 of these.

### 1.3.2. Bounding boxes

![table](fig4.png)

For the training set, we annotated boxes in 1.74M images, for the available positive human-verified image-level labels. We focused on the most specific labels. For example, if an image has labels {car, limousine, screwdriver}, we annotated boxes for limousine and screwdriver. For each positive label in an image, we exhaustively annotated every instance of that object class in the image (but see below for group cases). We provide 14.6M bounding boxes. On average there are 8.4 boxed objects per image. 90% of the boxes were manually drawn by professional annotators at Google using the efficient extreme clicking interface [1] (new in V6: the actual four extreme points defining each box are released for train). We produced the remaining 10% semi-automatically using an enhanced version of the method in [2]. These boxes have been human verified to have IoU>0.7 with a perfect box on the object, and in practice they are accurate (mean IoU ~0.77, see Sect. 4.2 of [3]). We have drawn bounding boxes for human body parts and the class "Mammal" only for 95,335 images, due to the overwhelming number of instances (1,327,596 on the 95,335 images). This list of images enables using the data correctly during training of object detectors (as there might be a positive image label for a human body part, and yet no boxes). Finally, we drew a single box around groups of objects (e.g., a bed of flowers or a crowd of people) if they had more than 5 instances which were heavily occluding each other and were physically touching (we marked these boxes with the attribute "group-of").

For the validation and test sets, we provide exhaustive box annotation for all object instances, for all available positive image-level labels (again, except for "groups-of"). All boxes were manually drawn. We deliberately tried to annotate boxes at the most specific level in our semantic hierarchy as possible. On average, there are 7.4 boxes per image in the validation and test sets. For Open Images V5, we improved the annotation density, which now comes close to the density in the training set. This ensures more precise evaluation of object detection models. In contrast to the training set, on the validation and test sets we annotated human body parts on all images for which we have a positive label.

We emphasize that the images are annotated both human-verified positive and negative labels (see section above). Importantly, the negative image-level labels can be used during training of object detectors, e.g. for hard-negative mining. Moreover, they can also be used during evaluation, as detections of a class annotated as negative (absent) in the ground-truth can be reliably counted as false-positives. In our Open Images Challenge website we present an evaluation metric that fully uses the image-level labels to fairly evaluate detection models.

In all splits (train, val, test), annotators also marked a set of attributes for each box, e.g. indicating whether that object is occluded (see the full description in the download section).

### 1.3.3. Object segmentations

Table 3 shows an overview of the object segmentation annotations in all splits of the dataset. These annotations cover a subset of 350 classes from the 600 boxed classes. These offer a broader range of categories than Cityscapes or COCO, and cover more images and instances than ADE20k. The segmentations are spread over a subset of the images with bounding boxes (Table 2).

![table](fig5.png)

For the training set we annotated 2.7M instance masks, starting from the available bounding boxes. The masks cover 350 classes and are spread over 944k images. On average there are 2.8 segmented instances per image. The segmentation masks on the training set have been produced by a state-of-the-art interactive segmentation process , where professional human annotators iteratively correct the output of a segmentation neural network. This is more efficient than manual drawing alone, while at the same time delivering accurate masks (mIoU 84% ).

We selected the 350 classes to annotate with segmentation masks based on the following criteria: (1) whether the class exhibits one coherent appearance over which a policy could be defined (e.g. "hiking equipment" is rather ill-defined); (2) whether a clear annotation policy can be defined (e.g. which pixels belong to a nose?); and (3) whether we expect current segmentation neural networks to be able to capture the shape of the class adequately (e.g. jellyfish contains thin structures that are hard for state-of-the-art models). We have put particular effort into ensuring consistent annotations across different objects (e.g., all cat masks include their tail; bags carried by camels or persons, are included in their mask).

We annotated all boxed instances of these 350 classes on the training split that fulfill the following criteria: (1) the object size is larger than 40x80 or 80x40 pixels; (2) the object boundaries can be confidently determined by the annotator (e.g. blurry or very dark instances are skipped); (3) the bounding-box contains a single real object (i.e. does not have any of the IsGroupOf, IsDepiction, IsInside attributes). A few of the 350 classes have a disproportionately large number of instances. To better spread the annotation effort we capped four categories: "clothing" to 441k instances, "person" to 149k, "woman" to 117k, "man" to 114k. In total we annotated segmentation masks for 769k instances of "person"+"man"+"woman"+"boy"+"girl"+"human body". All other classes are annotated without caps, using only the two criteria above.
The per-instance corrective clicks generated by the interactive segmentation process are included in the mask data files, and as part of the point-level labels (see below).

For the validation and test splits we created 99k masks spread over 54k images. These have been annotated with a purely manual free-painting tool and with a strong focus on quality. They are near-perfect (self-consistency 90% mIoU) and capture even fine details of complex object boundaries (e.g. spiky flowers and thin structures in man-made objects). For the validation and test splits we limited these annotation to a maximum of 600 instances per class (per split), and applied the same instance selection criteria as in the training split (minimal size, unambiguous boundary, single real object). On average over all instances, both our training and validation+test annotations offer more accurate object boundaries than the polygon annotations provided by most existing datasets.

Please note that instances without a mask remain covered by their corresponding bounding boxes, and thus can be appropriately handled during training and evaluation of segmentation models.

We emphasize that the images are annotated both human-verified positive and negative labels. The negative image-level labels can be used during training of segmentation models, e.g. for hard-negative mining. Moreover, they can also be used during evaluation, as we do for the Open Images Challenge.

### 1.3.4. Visual relationships

Table 4 shows an overview of the visual relationship annotations in the dataset.

![table](fig6.png)

In our notation, a pair of objects connected by a relationship forms a triplet (e.g. "beer on table"). Visual attributes are also represented as triplets, where an object in connected with an attribute using the relationship is. We annotate two types of attributes: physical object properties (e.g. "table is wooden" or "handbag is made of leather") and human actions (e.g. "man is jumping" or "woman is standing"). We initially selected 2019 possible triplets based on existing bounding box annotations. The 1,466 of them that have at least one instance in the training split form the final set of visual relationships/attributes triplets. In total, we annotated more than 3.1M instances of these triplets on the training split, involving 288 different object classes and 15 attributes. These include human-object relationships (e.g. "woman playing guitar", "man holding microphone"), object-object relationships (e.g. "beer on table", "dog inside car"), human-human relationships (e.g. "two men shake hands"), object attributes (e.g. "table is wooden"), and human actions (e.g. "man is jumping").

Visual relationship annotations are exhaustive (except human-human relationships, see next), meaning that for each image that can potentially contain a relationship triplet (i.e. that contains the objects involved in that triplet), we provide annotations exhaustively listing all positive triplets instances in that image. For example, for "woman playing guitar" in an image, we list all pairs of ("woman","guitar") that are in the relationship "playing" in that image. All other pairs of ("woman","guitar") in that image are reliable negative examples for the "playing" relationship. Further, human-human relationships were exhaustively annotated for the images that have the corresponding positively verified image-level label for the relationship (e.g., if an image has positively verified image-level label "hug", all pairs of people hugging would be annotated).

Finally, we annotated some zero-shot triplets: 61 distinct triplets in the validation and test sets do not have samples in the train set (and some triplets in train set do not have corresponding triplets on validation and test sets). Examples of these triplets are: "girl holds dumbbell", "pizza on a cutting board", or "dog on a washing machine".

We emphasize that the images are annotated both with human-verified positive and negative image-level labels (see section above). Importantly, the negative image-level labels can be used during training of visual relationship detectors: if any of the two object classes in a relationship triplet is marked as a negative label in our ground-truth, then all detections of that triplet are false-positives. The same can be done during evaluation, as we did for our official Open Images Challenge metric.

### 1.3.5. Localized narratives

Localized narratives are multimodal descriptions of images consisting of synchronized voice, text, and mouse traces over the objects being described. In Open Images V6 we released 675k localized narratives annotations on images from Open Images (Table 5).

![table](fig7.png)

Note we originally launched localized narratives only on train in V6, but since July 2020 we also have validation and test covered. More information about this type of annotations, as well as visualizations and annotations for other datasets can be found in the localized narratives standalone website.

### 1.3.6. Point-level labels

We provide class labels for large, diverse set of points over 1.4M images. These labels cover 5,827 classes including both things and stuff (e.g. things: dog, shoes; stuff: mountain, pavement). These point labels are suitable for zero/few-shot semantic segmentation training and evaluation. The number of points per class has a zipf-like distribution (few classes have many point labels, many classes have few labels).

![table](fig8.png)

The point labels were collected from human raters via three different methods:

PointVerification: A machine learning model generated questions of the kind "is this point on a pumpkin?". Human annotators then answered ‘yes’, or ‘no’, or ‘unsure’. Hence, we release both positive (‘yes’) and negative (‘no’) labels here.
CorrectiveClicks: Corrective clicks generated by the interactive segmentation process [4], converted into yes/no/unsure labels. We include the existing OpenImages V5 clicks, as well as many new ones.
FreeForm: Free-form clicking on the image + label input. These labels are always positive.
Table 6 shows some key statistics of the newly released data, and table 7 the grand total when including the converted v5 data.

Over the train+validation+test sets the 66M point-level class labels cover 57M unique points (image-id + xy coordinates).
5,180 classes have at least one yes point, 3,739 classes have 3 or more yes points. Out of the 5,827 known classes, 2,033 appear in all three train+validation+test sets.

![table](fig9.png)

Together FreeForm and PointVerification provide 5.6M yes point labels over 5,800 classes. In these two modalities, a class may be mentioned with multiple names (e.g. bottle & flask). We merge such cases via a named-entity recognition system into similar concepts (with an unique class identifier). The download page provides a list of such name variants ("Point-label classes metadata").

Compared to the v5 release, the v7 CorrectiveClicks release:

- Provide point-level semantic label (instead of instance segmentation in v5),
- Cover train, validation, and test sets (instead of train only in v5),
- Cover 39 new classes (389 total), and include 14.6M new points.

For CorrectiveClicks each point label corresponds to one human annotator click.
For FreeForm, each point label corresponds to one human annotator click plus a free text entry to name a class.
For PointVerification, point labels are deduced from multiple human votes over the generated yes-no questions. Unsure labels indicate votes that did not converge, or corrective clicks with ambiguous conversion.
Out of the overall 66M point labels, 40% are "yes" labels, 45% are "no" labels, and 15% are "unsure" labels. Details of each point-label vote are provided in the download files.

## 1.4. Class definitions

Classes are identified by MIDs (Machine-generated Ids) as can be found in Freebase or Google Knowledge Graph API. A short description of each class is available in class-descriptions.csv.

## 1.5. Preparing the dataset

We work on the validation set of the open images dataset, we only use the classes that overlap with coco dataset.

# Results

We evaluate each of the following models on the validation set of the open images dataset.

1. Faster R-CNN with ResNet-50
2. Faster R-CNN with MobileNet-V3
3. SSD

We will consider this image as a referrence

![img](canvas.png)

## Faster R-CNN ResNet-50

![img](fcnresnet50.png)

Evaluation time: 6.3 samples/sec

mAP : 0.0876

![table](resnet50/nothresh.png)

![plot](resnet50/plot.png)

![map](resnet50/map.png)

![plt](resnet50/thresh50/plt.png)

{'fn': 8375, 'tp': 1677, 'fp': 27439}

IOU Score 0.03565895646746864


### Using confidence threshold = 0.5

![table](resnet50/thresh50/map.png)

![avg](resnet50/thresh50/avg.png)

![plot](resnet50/thresh50/plot.png)

![plt](resnet50/thresh50/plt50.png)

accuracy :  0.09251302083333333

precision :  0.21117550899093476

recall :  0.1413649025069638

fscore :  0.16935820272927715

{'tp': 1421, 'fp': 5588, 'fn': 8631}

IOU Score 0.07371190296045621

### Using confidence threshold = 0.75

![map](resnet50/thresh75/map.png)

![avg](resnet50/thresh75/avg.png)

![plot](resnet50/thresh75/plot.png)

accuracy :  0.09589982438726426
precision :  0.2920251104394327
recall :  0.12495025865499403
fscore :  0.1750156761652616

![plt](resnet50/thresh75/plt.png)

{'fn': 8796, 'tp': 1256, 'fp': 3190}

IOU Score 0.07759311243309393

### Using confidence threshold = 0.9

![map](resnet50/thresh90/map.png)

![avg](resnet50/thresh90/avg.png)

![plot](resnet50/thresh90/plot.png)

accuracy :  0.08701902748414377
precision :  0.367237687366167
recall :  0.10236768802228412
fscore :  0.16010580364089

![plt](resnet50/thresh90/plt.png)

{'tp': 1029, 'fn': 9023, 'fp': 1839}

IOU Score 0.07176685677122789

## Faster R-CNN MobileNet-V3

Evaluation time: 14 samples/sec

![img](mobilenet.png)

### No threshold

![map](mobilenetv3/map.png)

![avg](mobilenetv3/avg.png)

![plot](mobilenetv3/plot.png)

![plt](mobilenetv3/plt.png)

accuracy :  0.06180111821086262

precision :  0.11035835264753076

recall :  0.12315957023477915

fscore :  0.11640808650681711

{'tp': 1029, None: 26248, 'fn': 9023, 'fp': 1839}

IOU Score 0.02237551309333415

### Using confidence threshold = 0.5

![map](mobilenetv3/thresh50/map.png)

![avg](mobilenetv3/thresh50/avg.png)

![plot](mobilenetv3/thresh50/plot.png)

![plt](mobilenetv3/thresh50/plt.png)

accuracy :  0.08462792893313878
precision :  0.38143133462282397
recall :  0.0980899323517708
fscore :  0.15604969533908367

{'fp': 1656, 'tp': 986, 'fn': 9066}

IOU Score 0.06987636876256666

### Using confidence threshold = 0.75

![map](mobilenetv3/thresh75/map.png)

![avg](mobilenetv3/thresh75/avg.png)

![plot](mobilenetv3/thresh75/plot.png)

![plt](mobilenetv3/thresh75/plt.png)

accuracy :  0.07731629392971245
precision :  0.484
recall :  0.08426183844011143
fscore :  0.14353499406880188

{'fn': 9205, 'tp': 847, 'fp': 925}

IOU Score 0.06511874350534122

### Using confidence threshold = 0.9


![map](mobilenetv3/thresh90/map.png)

![avg](mobilenetv3/thresh90/avg.png)

![plot](mobilenetv3/thresh90/plot.png)

![plt](mobilenetv3/thresh90/plt.png)

accuracy :  0.06494485813931568
precision :  0.5529695024077047
recall :  0.06854357341822523
fscore :  0.12196848999822978

{'fn': 9363, 'tp': 689, 'fp': 567}

IOU Score 0.05545436412649866

## SSD

Evaluation time: 4.8 samples/sec

![img](ssd41.png)

### Using confidence threshold = 0.1

![map](ssd/map.png)

![avg](ssd/avg.png)

![plot](ssd/plot.png)

![plt](ssd/plt.png)

accuracy :  0.041040518955569316
precision :  0.057530615975141655
recall :  0.12524870672502986
fscore :  0.07884519038076153

IOU Score 0.031539423830333736

{'fp': 22109, 'fn': 8793, 'tp': 1259}

### Using confidence threshold = 0.5

![map](ssd/thresh50/map.png)

![avg](ssd/thresh50/avg.png)

![plot](ssd/thresh50/plot.png)

![plt](ssd/thresh50/plt.png)

accuracy :  0.07298464932438643
precision :  0.48982109808760027
recall :  0.0789892558694787
fscore :  0.13604043519232414

{'fp': 842, 'fn': 9258, 'tp': 794}

IOU Score 0.06277767370556407

### Using confidence threshold = 0.75

![map](ssd/thresh75/map.png)

![avg](ssd/thresh75/avg.png)

![plot](ssd/thresh75/plot.png)

![plt](ssd/thresh75/plt.png)

'fn': 9415, 'tp': 637, 'fp': 464}

IOU Score 0.05322200448161114

### Using confidence threshold = 0.9

![map](ssd/thresh90/map.png)

![avg](ssd/thresh90/avg.png)

![plot](ssd/thresh90/plot.png)

![plt](ssd/thresh90/plt.png)

accuracy :  0.04587333915236155
precision :  0.6461748633879781
recall :  0.04705531237564664
fscore :  0.0877225519287834

{'fn': 9579, 'tp': 473, 'fp': 260}

IOU Score 0.040478422994867286
