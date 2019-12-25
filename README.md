# Arabic OCR

## setup

```shell
virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt
```


Project Pipeline
================

Prepossessing Module
--------------------

### Correcting Image Skew

first step here is to find a rectangle with the minimum area surrounding
all the text in the image in other words the rectangle containing all
the white pixels whilst having the smallest area, then the angle between
this rectangle and the original image is calculated and is considered to
be the skew of the image, this angle is further corrected if it has a
negative value, the angle value however, is implementation dependant,
then the image is skew corrected with this angle value.

![before skew
correction[]{label="fig:sub1"}](Figure/capr6.png){#fig:sub1
width=".4\\linewidth"}

![after skew
correction[]{label="fig:sub2"}](Figure/deskew.png){#fig:sub2
width=".4\\linewidth"}

Segmentation
------------

### Line segmentation

for line segmentation the horizontal projection of the text image is
calculated. that requires scanning the image's rows and counting the
number of black pixels. for text image of dimensions \[m,n\] where m is
the number of rows, n is the number of columns, the horizontal
projection is defined as: $$\begin{aligned}
H(i) &= \sum_{j=0}^{n-1} p(i,j) \\\end{aligned}$$

After obtaining the horizontal projection, a histogram analysis is
performed to find where the lowest values of pixel densities occur,
these are the places where the separation will take place.

![Horizontal Projection
[@Apostolos2013][]{label="fig:HorizontalProjection"}](Figure/horizontal_projection.png){#fig:HorizontalProjection
width="0.8\\linewidth"}

After experimenting with multiple images from the data set, it was found
that applying dilation with a (3,3) kernel enhanced the segmentation by
getting rid of empty line segments and making the all segments of
approximately the same size.

![Line Segments After
Dilation[]{label="fig:egmented lines"}](Figure/segmented_lines.png){#fig:egmented lines
width="0.8\\linewidth"}

### Word and sub-word segmentation

After lines segmentation, each line is then passed to the next level of
segmentation, the vertical projection of the text line is found and that
requires scanning image columns to each text line and counting the
number of black pixels. The vertical projection can be defined as:

$$\begin{aligned}
H(i) &= \sum_{j=0}^{n-1} p(i,j) \\\end{aligned}$$

![vertical projection for a line
image[]{label="fig:egmented lines"}](Figure/vertical_projection.png){#fig:egmented lines
width="0.8\\linewidth"}

for sub-word segmentation the same logic used in line segmentation can
be applied here, the places where the the vertical projection is equal
to zero are where the separation between sub-words occurs.

segmenting words, however is not as easy, first the baseline of each
line calculated, the baseline is the index of the highest value of the
horizontal projection of the line image .i.e the y coordinate with the
highest pixel density.

After calculating the baseline value, the distances between the words of
each line along the baseline are obtained then sorted according to their
frequency of occurrence, the threshold to consider the separating
distance between any two sub-words a word separating space is then set
to be the minimum value of the frequency sorted array of distances plus
the max value of the aforementioned array divided by 4, all distances is
then compared to that threshold if greater then it's considered a word
separating space.

### Character Segmentation

After trying out multiple approaches for character segmentation from
various papers and literature, we came up with our approach inspired by
all those failed experiments, this approach is heavily inspired by two
papers \[2\],\[3\]

first step is contour extraction, the method in the paper above relies
on extracting the upper contour of the sub-word in our case the quality
if the images was quit poor so separating upper and lower contour was
not a viable option so we worked on all of the contour.

a value we would call it local baseline is calculated from the contour
of each sub-word, not to be confused with the baseline of the image
line, it's the y coordinate with the highest frequency of occurrence,
it's considered to be the junction line of each line , this value should
be close to the baseline of the line image (within a range of 3 pixels)
if not the baseline of the line is used instead, using this value
instead of the baseline was found to improve the results.

then a loop on on all y coordinates of contour points, the point that
satisfy a certain criteria is saved along with their x coordinate as
candidate points, the criteria is that the y coordinate of the point is
within a 1 pixel range (above or below) from the baseline and this
condition is valid for more 2 consecutive points these points are saved
as candidate points each group of consecutive points is separation
region and only one point of each group is to be a final separation
point.

the list of candidate points for each region is then further processed
to eliminate false segmentation points, for each region the points that
doesn't satisfy the following conditions are eliminated:

-   the segment of the image between the baseline / 2 and baseline -1
    must have no white pixels i.e. empty.

-   the segment of the image between the baseline + 2 and the last
    column in the image must have no white pixels i.e. empty.

after filtration of points for each region the closest point to the
middle is then picked to be a segmentation point.

this method had problems with letters like DAL at the end of word, so an
additional check for the letter DAL at the last segmentation point is
made.

further processing is done to eliminate segments with no letters between
them by checking area in between the separation lines above the baseline
for white pixels if there is none then one of the lines is eliminated,
the one on the left is chosen.

this method is suffers from over segmentation with letters like SEEN and
SHEEN, to overcome this problem 2D convolution with a template for the
letter SEEN after removing the points to get matches with the SHEEN too.

Character Recognition
---------------------

For the recognition stage, we used the algorithm introduced in this
paper as reference, but we had to do some changes on the main flow of
the algorithm to get the best output on the dataset. for details, please
refer to the feature extraction subsection.

### Training

In Training, we need to find a set of prototype feature vectors for each
character, those vectors will be used as the basis for our recognition.
To find such vectors, we need to match labels from the dataset with
characters detected from previous phases, but we need to keep the margin
of error at min, so we filter out the data before matching using several
criteria on multiple levels:

-   Check if the number of detected words is the same as the number of
    words in the document, if not discard this whole image/text sample.

-   Check if the number of characters detected is the same as the
    corresponding word. If not, discard the current word.

-   Check if the characters segments represent valid characters or not.
    If not, discard the current word.

-   Check if the character segment posses the main characteristics of
    the character that it's being associated with. If not, then the
    current character segment is discarded.

-   If all checks passed, then the feature vector for this segment is
    associated with the corresponding character.

The output of this stage is a JSON formatted file where the main keys
are the scores and under each score is a set of tuples that contain the
character and it's associated feature vector, characters under the same
key have the same score for the associated feature vector.

### Prediction

As a first step, the JSON file containing the feature vector is loaded.
For each character-segmented word passed, each character segment is
evaluated to a feature vector and a score, the score is used to match
and find the set of characters that are most probably a good candidate
match for this segment.

In some cases the feature vector extracted from the segment is empty,
which in this case means there is no character in this segment, this
rejection criteria is core in effectively discarding empty segments and
correcting the number of expected characters to produced for the word
even though the character segmentation stage may have over-segmented.

If the feature vector found is not empty and the score was found in our
dataset, then we proceed to match this feature vector with one of the
feature vectors under the provided score, we choose the character whose
feature vector has the lowest euclidean distance to the target feature
vector.

Feature Extraction
==================

Selected Features
-----------------

The algorithm relies on feature extraction of each character, it focuses
on the features related to the Arabic characters and not just generic
features, those features include:

-   Existence of dots/hamza in the character(0/1).

-   Positioning of the dots/hamza in the character. (above, under or
    middle).

-   Number of dots (if any ) detected for this character.

-   Corner Variance: it's calculated from the corners of character
    matrix as follows the sum of weighted corners of the character
    matrix(1 if black pixel exists in this corner and 0 if
    not).\[Equation 1.1\]

-   Max Number of transitions in the x-axis: represents after sweeping
    the whole character horizontally, what was the max number of 0-1/1-0
    transitions met in one of the rows.

-   Max Number of transitions in the y-axis: represents after sweeping
    the whole character vertically, what was the max number of 0-1/1-0
    transitions met in one of the columns. $$\label{eq:Eadditional}
    			CorVar = \sum_{i} Corner_{i} \cdot  2^i$$

Score Calculation
-----------------

In addition to those features, we define a score for each character, the
score is calculated based on the characteristics of the character, using
concavities and holes to represent those characteristics\[Equation 1.2\]
$$\label{eq:score}
			score = H \cdot 2^0 + L  \cdot 2^1 +  R \cdot 2^2  + U \cdot 2^3 +  B \cdot 2^4$$
where: H: Number of holes in the character. L: Number of left
concavities in the character. R: Number of right concavities in the
character. B: Number of bottom concavities in the characters.

Interest Points
---------------

Concavities and holes are determined based on candidate essential
points, the essential points are in the middle of the distance between
the transitions for a row/column where the number of transitions is 4 or
higher. To determine if an essential point is a concavity or hole or
none of those, 8 beams are launched from all 8 neighbouring
directions(N,S,E,W,NE,NW,SE,SW). For an essential point to be a hole,
all beams need to find a blockage in their way, if only 3 consecutive
directions are blocked, then it's a concavity.

![lines where transitions are equal to or higher than 4 are candidates
for holes/concavities ](Figure/holes_concav.png){width="0.6\\linewidth"}

[\[fig:holes\_concavities\]]{#fig:holes_concavities
label="fig:holes_concavities"}

Experiments
===========

We made a number of trails either based on different papers or tweaks on
implemented algorithms, here we list all the trails we had before we
finally reach our current pipeline.

Character Segmentation using neural nets
----------------------------------------

In this experiment, we tried to design a neural net to replace the
character segmentation layer. The first problem that faced was the data,
how can we get labeled data that has every word image with its
associated character segments, so we approached this problem by using
our already implemented character segmentation layer to prepare the
date, and using our augmentation and character recognition to reject
wrongly segmented words. The promise behind this approach is that if we
could let a neural net learn from only the correct output of our
segmentation layer, then our network could generalize and learn to
recognize more words than the layer used to train it.

The second issue we faced was the output format, what is the most
effective way to represent the output, we tried a number of variants on
the output format and trained a neural net on each:

-   Input is an image and output is a line binary image having the same
    width as the original image with ones at position of segments and
    zeros elsewhere: this format landed a high accuracy upon training
    with a 1-2 RELU dense layers and a sigmoid, but although the
    accuracy reached 95 percent, we found that the network kind of
    cheated to get such accuracy as the number of ones or segments if
    very low compared to the number of zeros, so the network was much
    more interested in setting the zeros right than the ones, we needed
    a more loss function that lays higher cost upon missing the segments
    positions but it should be back-propagate-able too.

-   Input is an image and output is an array of 14 values where each
    value either represents an index to segment position or 0 which
    means no segment. This approach got around 78-77 accuracy, but the
    numbers kind dense layers of drifted and the words were too
    sensitive to such shifts in segments. The architecture for this
    network was 2 RELU layers with flattening to get 14 values out the
    2d matrix in the previous layer.

-   Input is an image and output is an line array with the same width as
    the original image, each element in the array contains a number that
    represents the label of its segment(.i.e. 111112222233334444), this
    format made a very poor accuracy using the same architecture as the
    first one with accuracies below 50 percent

Enhancements and Future work
============================

for character segmentation a more accurate methods for extracting upper
contour can be further investigated and incorporated to enhance the
quality and performance speed of the character segmentation.

References
==========

1.  [Recognition system for printed multi-font and multi-size Arabic
    characters, April 2002, Latifa Hamami Daoud- Berkani Daoud
    Berkani](https://www.researchgate.net/publication/250734751_Recognition_system_for_printed_multi-font_and_multi-size_Arabic_characters).

2.  [A new hybrid method for Arabic multi-font text segmentation, and a
    reference corpus construction, Abdelhay Zoizou, Arsalane, Zarghili,
    Ilham
    Chaker](https://www.researchgate.net/publication/326331801_A_new_hybrid_method_for_Arabic_Multi-font_text_segmentation_and_a_reference_corpus_constr  uction)

3.  [Contour-based character segmentation for printed Arabic text with
    diacritics, Sos Agaian, Khader Mohammad, Aziz
    Qaroush](https://www.researchgate.net/publication/335506453_Contour-based_character_segmentation_for_printed_Arabic_text_with_diacritics)