import glob
import json
from collections import Counter
import logging
import os
import numpy as np
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/logs/hyperopt/RAPTARCHIS_FLAT_XGBOOST_hyperopt.txt',
                    filemode='a')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

LOGGER = logging.getLogger(__name__)

greek_stopwords = ['Α∆ΙΑΚΟΠΑ', 'ΑΙ', 'ΑΚΟΜΑ', 'ΑΚΟΜΗ', 'ΑΚΡΙΒΩΣ', 'ΑΛΗΘΕΙΑ', 'ΑΛΗΘΙΝΑ', 'ΑΛΛΑ', 'ΑΛΛΑΧΟΥ', 'ΑΛΛΕΣ', 'ΑΛΛΗ', 'ΑΛΛΗΝ', 'ΑΛΛΗΣ',
                   'ΑΛΛΙΩΣ', 'ΑΛΛΙΩΤΙΚΑ', 'ΑΛΛΟ', 'ΑΛΛΟΙ', 'ΑΛΛΟΙΩΣ', 'ΑΛΛΟΙΩΤΙΚΑ', 'ΑΛΛΟΝ', 'ΑΛΛΟΣ', 'ΑΛΛΟΤΕ', 'ΑΛΛΟΥ', 'ΑΛΛΟΥΣ', 'ΑΛΛΩΝ', 'ΑΜΑ',
                   'ΑΜΕΣΑ', 'ΑΜΕΣΩΣ', 'ΑΝ', 'ΑΝΑ', 'ΑΝΑΜΕΣΑ', 'ΑΝΑΜΕΤΑΞΥ', 'ΑΝΕΥ', 'ΑΝΤΙ', 'ΑΝΤΙΠΕΡΑ', 'ΑΝΤΙΣ', 'ΑΝΩ', 'ΑΝΩΤΕΡΩ', 'ΑΞΑΦΝΑ', 'ΑΠ',
                   'ΑΠΕΝΑΝΤΙ', 'ΑΠΟ', 'ΑΠΟΨΕ', 'ΑΡΑ', 'ΑΡΑΓΕ', 'ΑΡΓΑ', 'ΑΡΓΟΤΕΡΟ', 'ΑΡΙΣΤΕΡΑ', 'ΑΡΚΕΤΑ', 'ΑΡΧΙΚΑ', 'ΑΣ', 'ΑΥΡΙΟ', 'ΑΥΤΑ', 'ΑΥΤΕΣ',
                   'ΑΥΤΗ', 'ΑΥΤΗΝ', 'ΑΥΤΗΣ', 'ΑΥΤΟ', 'ΑΥΤΟΙ', 'ΑΥΤΟΝ', 'ΑΥΤΟΣ', 'ΑΥΤΟΥ', 'ΑΥΤΟΥΣ', 'ΑΥΤΩΝ', 'ΑΦΟΤΟΥ', 'ΑΦΟΥ', 'ΒΕΒΑΙΑ', 'ΒΕΒΑΙΟΤΑΤΑ',
                   'ΓΙ', 'ΓΙΑ', 'ΓΡΗΓΟΡΑ', 'ΓΥΡΩ', '∆Α', '∆Ε', '∆ΕΙΝΑ', '∆ΕΝ', '∆ΕΞΙΑ', '∆ΗΘΕΝ', '∆ΗΛΑ∆Η', '∆Ι', '∆ΙΑ', '∆ΙΑΡΚΩΣ', '∆ΙΚΑ', '∆ΙΚΟ',
                   '∆ΙΚΟΙ', '∆ΙΚΟΣ', '∆ΙΚΟΥ', '∆ΙΚΟΥΣ', '∆ΙΟΛΟΥ', '∆ΙΠΛΑ', '∆ΙΧΩΣ', 'ΕΑΝ', 'ΕΑΥΤΟ', 'ΕΑΥΤΟΝ', 'ΕΑΥΤΟΥ', 'ΕΑΥΤΟΥΣ', 'ΕΑΥΤΩΝ',
                   'ΕΓΚΑΙΡΑ', 'ΕΓΚΑΙΡΩΣ', 'ΕΓΩ', 'Ε∆Ω', 'ΕΙ∆ΕΜΗ', 'ΕΙΘΕ', 'ΕΙΜΑΙ', 'ΕΙΜΑΣΤΕ', 'ΕΙΝΑΙ', 'ΕΙΣ', 'ΕΙΣΑΙ', 'ΕΙΣΑΣΤΕ', 'ΕΙΣΤΕ', 'ΕΙΤΕ',
                   'ΕΙΧΑ', 'ΕΙΧΑΜΕ', 'ΕΙΧΑΝ', 'ΕΙΧΑΤΕ', 'ΕΙΧΕ', 'ΕΙΧΕΣ', 'ΕΚΑΣΤΑ', 'ΕΚΑΣΤΕΣ', 'ΕΚΑΣΤΗ', 'ΕΚΑΣΤΗΝ', 'ΕΚΑΣΤΗΣ', 'ΕΚΑΣΤΟ', 'ΕΚΑΣΤΟΙ',
                   'ΕΚΑΣΤΟΝ', 'ΕΚΑΣΤΟΣ', 'ΕΚΑΣΤΟΥ', 'ΕΚΑΣΤΟΥΣ', 'ΕΚΑΣΤΩΝ', 'ΕΚΕΙ', 'ΕΚΕΙΝΑ', 'ΕΚΕΙΝΕΣ', 'ΕΚΕΙΝΗ', 'ΕΚΕΙΝΗΝ', 'ΕΚΕΙΝΗΣ', 'ΕΚΕΙΝΟ',
                   'ΕΚΕΙΝΟΙ', 'ΕΚΕΙΝΟΝ', 'ΕΚΕΙΝΟΣ', 'ΕΚΕΙΝΟΥ', 'ΕΚΕΙΝΟΥΣ', 'ΕΚΕΙΝΩΝ', 'ΕΚΤΟΣ', 'ΕΜΑΣ', 'ΕΜΕΙΣ', 'ΕΜΕΝΑ', 'ΕΜΠΡΟΣ', 'ΕΝ', 'ΕΝΑ',
                   'ΕΝΑΝ', 'ΕΝΑΣ', 'ΕΝΟΣ', 'ΕΝΤΕΛΩΣ', 'ΕΝΤΟΣ', 'ΕΝΤΩΜΕΤΑΞΥ', 'ΕΝΩ', 'ΕΞ', 'ΕΞΑΦΝΑ', 'ΕΞΗΣ', 'ΕΞΙΣΟΥ', 'ΕΞΩ', 'ΕΠΑΝΩ', 'ΕΠΕΙ∆Η',
                   'ΕΠΕΙΤΑ', 'ΕΠΙ', 'ΕΠΙΣΗΣ', 'ΕΠΟΜΕΝΩΣ', 'ΕΣΑΣ', 'ΕΣΕΙΣ', 'ΕΣΕΝΑ', 'ΕΣΤΩ', 'ΕΣΥ', 'ΕΤΕΡΑ', 'ΕΤΕΡΑΙ', 'ΕΤΕΡΑΣ', 'ΕΤΕΡΕΣ', 'ΕΤΕΡΗ',
                   'ΕΤΕΡΗΣ', 'ΕΤΕΡΟ', 'ΕΤΕΡΟΙ', 'ΕΤΕΡΟΝ', 'ΕΤΕΡΟΣ', 'ΕΤΕΡΟΥ', 'ΕΤΕΡΟΥΣ', 'ΕΤΕΡΩΝ', 'ΕΤΟΥΤΑ', 'ΕΤΟΥΤΕΣ', 'ΕΤΟΥΤΗ', 'ΕΤΟΥΤΗΝ',
                   'ΕΤΟΥΤΗΣ', 'ΕΤΟΥΤΟ', 'ΕΤΟΥΤΟΙ', 'ΕΤΟΥΤΟΝ', 'ΕΤΟΥΤΟΣ', 'ΕΤΟΥΤΟΥ', 'ΕΤΟΥΤΟΥΣ', 'ΕΤΟΥΤΩΝ', 'ΕΤΣΙ', 'ΕΥΓΕ', 'ΕΥΘΥΣ', 'ΕΥΤΥΧΩΣ',
                   'ΕΦΕΞΗΣ', 'ΕΧΕΙ', 'ΕΧΕΙΣ', 'ΕΧΕΤΕ', 'ΕΧΘΕΣ', 'ΕΧΟΜΕ', 'ΕΧΟΥΜΕ', 'ΕΧΟΥΝ', 'ΕΧΤΕΣ', 'ΕΧΩ', 'ΕΩΣ', 'Η', 'Η∆Η', 'ΗΜΑΣΤΑΝ', 'ΗΜΑΣΤΕ',
                   'ΗΜΟΥΝ', 'ΗΣΑΣΤΑΝ', 'ΗΣΑΣΤΕ', 'ΗΣΟΥΝ', 'ΗΤΑΝ', 'ΗΤΑΝΕ', 'ΗΤΟΙ', 'ΗΤΤΟΝ', 'ΘΑ', 'Ι', 'Ι∆ΙΑ', 'Ι∆ΙΑΝ', 'Ι∆ΙΑΣ', 'Ι∆ΙΕΣ', 'Ι∆ΙΟ',
                   'Ι∆ΙΟΙ', 'Ι∆ΙΟΝ', 'Ι∆ΙΟΣ', 'Ι∆ΙΟΥ', 'Ι∆ΙΟΥΣ', 'Ι∆ΙΩΝ', 'Ι∆ΙΩΣ', 'ΙΙ', 'ΙΙΙ', 'ΙΣΑΜΕ', 'ΙΣΙΑ', 'ΙΣΩΣ', 'ΚΑΘΕ', 'ΚΑΘΕΜΙΑ',
                   'ΚΑΘΕΜΙΑΣ', 'ΚΑΘΕΝΑ', 'ΚΑΘΕΝΑΣ', 'ΚΑΘΕΝΟΣ', 'ΚΑΘΕΤΙ', 'ΚΑΘΟΛΟΥ', 'ΚΑΘΩΣ', 'ΚΑΙ', 'ΚΑΚΑ', 'ΚΑΚΩΣ', 'ΚΑΛΑ', 'ΚΑΛΩΣ', 'ΚΑΜΙΑ',
                   'ΚΑΜΙΑΝ', 'ΚΑΜΙΑΣ', 'ΚΑΜΠΟΣΑ', 'ΚΑΜΠΟΣΕΣ', 'ΚΑΜΠΟΣΗ', 'ΚΑΜΠΟΣΗΝ', 'ΚΑΜΠΟΣΗΣ', 'ΚΑΜΠΟΣΟ', 'ΚΑΜΠΟΣΟΙ', 'ΚΑΜΠΟΣΟΝ', 'ΚΑΜΠΟΣΟΣ',
                   'ΚΑΜΠΟΣΟΥ', 'ΚΑΜΠΟΣΟΥΣ', 'ΚΑΜΠΟΣΩΝ', 'ΚΑΝΕΙΣ', 'ΚΑΝΕΝ', 'ΚΑΝΕΝΑ', 'ΚΑΝΕΝΑΝ', 'ΚΑΝΕΝΑΣ', 'ΚΑΝΕΝΟΣ', 'ΚΑΠΟΙΑ', 'ΚΑΠΟΙΑΝ', 'ΚΑΠΟΙΑΣ',
                   'ΚΑΠΟΙΕΣ', 'ΚΑΠΟΙΟ', 'ΚΑΠΟΙΟΙ', 'ΚΑΠΟΙΟΝ', 'ΚΑΠΟΙΟΣ', 'ΚΑΠΟΙΟΥ', 'ΚΑΠΟΙΟΥΣ', 'ΚΑΠΟΙΩΝ', 'ΚΑΠΟΤΕ', 'ΚΑΠΟΥ', 'ΚΑΠΩΣ', 'ΚΑΤ', 'ΚΑΤΑ',
                   'ΚΑΤΙ', 'ΚΑΤΙΤΙ', 'ΚΑΤΟΠΙΝ', 'ΚΑΤΩ', 'ΚΙΟΛΑΣ', 'ΚΛΠ', 'ΚΟΝΤΑ', 'ΚΤΛ', 'ΚΥΡΙΩΣ', 'ΛΙΓΑΚΙ', 'ΛΙΓΟ', 'ΛΙΓΩΤΕΡΟ', 'ΛΟΓΩ', 'ΛΟΙΠΑ',
                   'ΛΟΙΠΟΝ', 'ΜΑ', 'ΜΑΖΙ', 'ΜΑΚΑΡΙ', 'ΜΑΚΡΥΑ', 'ΜΑΛΙΣΤΑ', 'ΜΑΛΛΟΝ', 'ΜΑΣ', 'ΜΕ', 'ΜΕΘΑΥΡΙΟ', 'ΜΕΙΟΝ', 'ΜΕΛΕΙ', 'ΜΕΛΛΕΤΑΙ', 'ΜΕΜΙΑΣ',
                   'ΜΕΝ', 'ΜΕΡΙΚΑ', 'ΜΕΡΙΚΕΣ', 'ΜΕΡΙΚΟΙ', 'ΜΕΡΙΚΟΥΣ', 'ΜΕΡΙΚΩΝ', 'ΜΕΣΑ', 'ΜΕΤ', 'ΜΕΤΑ', 'ΜΕΤΑΞΥ', 'ΜΕΧΡΙ', 'ΜΗ', 'ΜΗ∆Ε', 'ΜΗΝ',
                   'ΜΗΠΩΣ', 'ΜΗΤΕ', 'ΜΙΑ', 'ΜΙΑΝ', 'ΜΙΑΣ', 'ΜΟΛΙΣ', 'ΜΟΛΟΝΟΤΙ', 'ΜΟΝΑΧΑ', 'ΜΟΝΕΣ', 'ΜΟΝΗ', 'ΜΟΝΗΝ', 'ΜΟΝΗΣ', 'ΜΟΝΟ', 'ΜΟΝΟΙ',
                   'ΜΟΝΟΜΙΑΣ', 'ΜΟΝΟΣ', 'ΜΟΝΟΥ', 'ΜΟΝΟΥΣ', 'ΜΟΝΩΝ', 'ΜΟΥ', 'ΜΠΟΡΕΙ', 'ΜΠΟΡΟΥΝ', 'ΜΠΡΑΒΟ', 'ΜΠΡΟΣ', 'ΝΑ', 'ΝΑΙ', 'ΝΩΡΙΣ', 'ΞΑΝΑ',
                   'ΞΑΦΝΙΚΑ', 'Ο', 'ΟΙ', 'ΟΛΑ', 'ΟΛΕΣ', 'ΟΛΗ', 'ΟΛΗΝ', 'ΟΛΗΣ', 'ΟΛΟ', 'ΟΛΟΓΥΡΑ', 'ΟΛΟΙ', 'ΟΛΟΝ', 'ΟΛΟΝΕΝ', 'ΟΛΟΣ', 'ΟΛΟΤΕΛΑ', 'ΟΛΟΥ',
                   'ΟΛΟΥΣ', 'ΟΛΩΝ', 'ΟΛΩΣ', 'ΟΛΩΣ∆ΙΟΛΟΥ', 'ΟΜΩΣ', 'ΟΠΟΙΑ', 'ΟΠΟΙΑ∆ΗΠΟΤΕ', 'ΟΠΟΙΑΝ', 'ΟΠΟΙΑΝ∆ΗΠΟΤΕ', 'ΟΠΟΙΑΣ', 'ΟΠΟΙΑΣ∆ΗΠΟΤΕ',
                   'ΟΠΟΙ∆ΗΠΟΤΕ', 'ΟΠΟΙΕΣ', 'ΟΠΟΙΕΣ∆ΗΠΟΤΕ', 'ΟΠΟΙΟ', 'ΟΠΟΙΟ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΙ', 'ΟΠΟΙΟΝ', 'ΟΠΟΙΟΝ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΣ', 'ΟΠΟΙΟΣ∆ΗΠΟΤΕ',
                   'ΟΠΟΙΟΥ', 'ΟΠΟΙΟΥ∆ΗΠΟΤΕ', 'ΟΠΟΙΟΥΣ', 'ΟΠΟΙΟΥΣ∆ΗΠΟΤΕ', 'ΟΠΟΙΩΝ', 'ΟΠΟΙΩΝ∆ΗΠΟΤΕ', 'ΟΠΟΤΕ', 'ΟΠΟΤΕ∆ΗΠΟΤΕ', 'ΟΠΟΥ', 'ΟΠΟΥ∆ΗΠΟΤΕ',
                   'ΟΠΩΣ', 'ΟΡΙΣΜΕΝΑ', 'ΟΡΙΣΜΕΝΕΣ', 'ΟΡΙΣΜΕΝΩΝ', 'ΟΡΙΣΜΕΝΩΣ', 'ΟΣΑ', 'ΟΣΑ∆ΗΠΟΤΕ', 'ΟΣΕΣ', 'ΟΣΕΣ∆ΗΠΟΤΕ', 'ΟΣΗ', 'ΟΣΗ∆ΗΠΟΤΕ', 'ΟΣΗΝ',
                   'ΟΣΗΝ∆ΗΠΟΤΕ', 'ΟΣΗΣ', 'ΟΣΗΣ∆ΗΠΟΤΕ', 'ΟΣΟ', 'ΟΣΟ∆ΗΠΟΤΕ', 'ΟΣΟΙ', 'ΟΣΟΙ∆ΗΠΟΤΕ', 'ΟΣΟΝ', 'ΟΣΟΝ∆ΗΠΟΤΕ', 'ΟΣΟΣ', 'ΟΣΟΣ∆ΗΠΟΤΕ', 'ΟΣΟΥ',
                   'ΟΣΟΥ∆ΗΠΟΤΕ', 'ΟΣΟΥΣ', 'ΟΣΟΥΣ∆ΗΠΟΤΕ', 'ΟΣΩΝ', 'ΟΣΩΝ∆ΗΠΟΤΕ', 'ΟΤΑΝ', 'ΟΤΙ', 'ΟΤΙ∆ΗΠΟΤΕ', 'ΟΤΟΥ', 'ΟΥ', 'ΟΥ∆Ε', 'ΟΥΤΕ', 'ΟΧΙ',
                   'ΠΑΛΙ', 'ΠΑΝΤΟΤΕ', 'ΠΑΝΤΟΥ', 'ΠΑΝΤΩΣ', 'ΠΑΡΑ', 'ΠΕΡΑ', 'ΠΕΡΙ', 'ΠΕΡΙΠΟΥ', 'ΠΕΡΙΣΣΟΤΕΡΟ', 'ΠΕΡΣΙ', 'ΠΕΡΥΣΙ', 'ΠΙΑ', 'ΠΙΘΑΝΟΝ',
                   'ΠΙΟ', 'ΠΙΣΩ', 'ΠΛΑΙ', 'ΠΛΕΟΝ', 'ΠΛΗΝ', 'ΠΟΙΑ', 'ΠΟΙΑΝ', 'ΠΟΙΑΣ', 'ΠΟΙΕΣ', 'ΠΟΙΟ', 'ΠΟΙΟΙ', 'ΠΟΙΟΝ', 'ΠΟΙΟΣ', 'ΠΟΙΟΥ', 'ΠΟΙΟΥΣ',
                   'ΠΟΙΩΝ', 'ΠΟΛΥ', 'ΠΟΣΕΣ', 'ΠΟΣΗ', 'ΠΟΣΗΝ', 'ΠΟΣΗΣ', 'ΠΟΣΟΙ', 'ΠΟΣΟΣ', 'ΠΟΣΟΥΣ', 'ΠΟΤΕ', 'ΠΟΥ', 'ΠΟΥΘΕ', 'ΠΟΥΘΕΝΑ', 'ΠΡΕΠΕΙ',
                   'ΠΡΙΝ', 'ΠΡΟ', 'ΠΡΟΚΕΙΜΕΝΟΥ', 'ΠΡΟΚΕΙΤΑΙ', 'ΠΡΟΠΕΡΣΙ', 'ΠΡΟΣ', 'ΠΡΟΤΟΥ', 'ΠΡΟΧΘΕΣ', 'ΠΡΟΧΤΕΣ', 'ΠΡΩΤΥΤΕΡΑ', 'ΠΩΣ', 'ΣΑΝ', 'ΣΑΣ',
                   'ΣΕ', 'ΣΕΙΣ', 'ΣΗΜΕΡΑ', 'ΣΙΓΑ', 'ΣΟΥ', 'ΣΤΑ', 'ΣΤΗ', 'ΣΤΗΝ', 'ΣΤΗΣ', 'ΣΤΙΣ', 'ΣΤΟ', 'ΣΤΟΝ', 'ΣΤΟΥ', 'ΣΤΟΥΣ', 'ΣΤΩΝ', 'ΣΥΓΧΡΟΝΩΣ',
                   'ΣΥΝ', 'ΣΥΝΑΜΑ', 'ΣΥΝΕΠΩΣ', 'ΣΥΝΗΘΩΣ', 'ΣΥΧΝΑ', 'ΣΥΧΝΑΣ', 'ΣΥΧΝΕΣ', 'ΣΥΧΝΗ', 'ΣΥΧΝΗΝ', 'ΣΥΧΝΗΣ', 'ΣΥΧΝΟ', 'ΣΥΧΝΟΙ', 'ΣΥΧΝΟΝ',
                   'ΣΥΧΝΟΣ', 'ΣΥΧΝΟΥ', 'ΣΥΧΝΟΥ', 'ΣΥΧΝΟΥΣ', 'ΣΥΧΝΩΝ', 'ΣΥΧΝΩΣ', 'ΣΧΕ∆ΟΝ', 'ΣΩΣΤΑ', 'ΤΑ', 'ΤΑ∆Ε', 'ΤΑΥΤΑ', 'ΤΑΥΤΕΣ', 'ΤΑΥΤΗ', 'ΤΑΥΤΗΝ',
                   'ΤΑΥΤΗΣ', 'ΤΑΥΤΟ,ΤΑΥΤΟΝ', 'ΤΑΥΤΟΣ', 'ΤΑΥΤΟΥ', 'ΤΑΥΤΩΝ', 'ΤΑΧΑ', 'ΤΑΧΑΤΕ', 'ΤΕΛΙΚΑ', 'ΤΕΛΙΚΩΣ', 'ΤΕΣ', 'ΤΕΤΟΙΑ', 'ΤΕΤΟΙΑΝ',
                   'ΤΕΤΟΙΑΣ', 'ΤΕΤΟΙΕΣ', 'ΤΕΤΟΙΟ', 'ΤΕΤΟΙΟΙ', 'ΤΕΤΟΙΟΝ', 'ΤΕΤΟΙΟΣ', 'ΤΕΤΟΙΟΥ', 'ΤΕΤΟΙΟΥΣ', 'ΤΕΤΟΙΩΝ', 'ΤΗ', 'ΤΗΝ', 'ΤΗΣ', 'ΤΙ',
                   'ΤΙΠΟΤΑ', 'ΤΙΠΟΤΕ', 'ΤΙΣ', 'ΤΟ', 'ΤΟΙ', 'ΤΟΝ', 'ΤΟΣ', 'ΤΟΣΑ', 'ΤΟΣΕΣ', 'ΤΟΣΗ', 'ΤΟΣΗΝ', 'ΤΟΣΗΣ', 'ΤΟΣΟ', 'ΤΟΣΟΙ', 'ΤΟΣΟΝ', 'ΤΟΣΟΣ',
                   'ΤΟΣΟΥ', 'ΤΟΣΟΥΣ', 'ΤΟΣΩΝ', 'ΤΟΤΕ', 'ΤΟΥ', 'ΤΟΥΛΑΧΙΣΤΟ', 'ΤΟΥΛΑΧΙΣΤΟΝ', 'ΤΟΥΣ', 'ΤΟΥΤΑ', 'ΤΟΥΤΕΣ', 'ΤΟΥΤΗ', 'ΤΟΥΤΗΝ', 'ΤΟΥΤΗΣ',
                   'ΤΟΥΤΟ', 'ΤΟΥΤΟΙ', 'ΤΟΥΤΟΙΣ', 'ΤΟΥΤΟΝ', 'ΤΟΥΤΟΣ', 'ΤΟΥΤΟΥ', 'ΤΟΥΤΟΥΣ', 'ΤΟΥΤΩΝ', 'ΤΥΧΟΝ', 'ΤΩΝ', 'ΤΩΡΑ', 'ΥΠ', 'ΥΠΕΡ', 'ΥΠΟ',
                   'ΥΠΟΨΗ', 'ΥΠΟΨΙΝ', 'ΥΣΤΕΡΑ', 'ΦΕΤΟΣ', 'ΧΑΜΗΛΑ', 'ΧΘΕΣ', 'ΧΤΕΣ', 'ΧΩΡΙΣ', 'ΧΩΡΙΣΤΑ', 'ΨΗΛΑ', 'Ω', 'ΩΡΑΙΑ', 'ΩΣ', 'ΩΣΑΝ', 'ΩΣΟΤΟΥ',
                   'ΩΣΠΟΥ', 'ΩΣΤΕ', 'ΩΣΤΟΣΟ', 'ΩΧ']

LOGGER.info('Load labels\' data')
LOGGER.info('-------------------')

# Load train dataset and count labels
train_files = glob.glob('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/train/*.json')
train_counts = Counter()
for filename in tqdm.tqdm(train_files):
    with open(filename) as file:
        data = json.load(file)
        train_counts[data['volume']] += 1

train_concepts = set(list(train_counts))

frequent, few = [], []
for i, (label, count) in enumerate(train_counts.items()):
    if count > 50:
        frequent.append(label)
    else:
        few.append(label)

# Load dev/test datasets and count labels
rest_files = glob.glob('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/dev/*.json')
rest_files += glob.glob('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/test/*.json')
rest_concepts = set()
for filename in tqdm.tqdm(rest_files):
    with open(filename) as file:
        data = json.load(file)
        rest_concepts.add(data['volume'])

# Compute zero-shot group
zero = list(rest_concepts.difference(train_concepts))

label_ids = dict()
margins = [(0, len(frequent) + len(few) + len(zero))]
k = 0
for group in [frequent, few, zero]:
    margins.append((k, k + len(group)))
    for concept in group:
        label_ids[concept] = k
        k += 1
margins[-1] = (margins[-1][0], len(frequent) + len(few) + len(zero))

# Load label descriptors
with open('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/raptarchis_el.json') as file:
    data = json.load(file)

LOGGER.info('Frequent labels: {}'.format(len(frequent)))
LOGGER.info('Few labels:      {}'.format(len(few)))
LOGGER.info('Zero labels:     {}'.format(len(zero)))


def parse_dataset(dataset, one_hot=False):
    x = []
    filenames = glob.glob(dataset)
    if one_hot is True:
        y = np.zeros((len(filenames), len(label_ids)), dtype=np.int32)
    else:
        y = np.zeros((len(filenames), ), dtype=np.int32)
    for i, filename in enumerate(filenames):
        with open(filename) as file:
            data = json.load(file)
            x.append(' '.join(data['tokens']))
            if one_hot is True:
                y[i][label_ids[data['volume']]] = 1
            else:
                y[i] = label_ids[data['volume']]
    return np.asarray(x), y


def mean_precision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_precision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


def mean_rprecision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / k


def ranking_rprecision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def probas_to_classes(probabilities):
    predictions = np.zeros(probabilities.shape, dtype=np.int32)
    for i, pred in enumerate(probabilities.argmax(axis=-1)):
        predictions[i][pred] = 1
    return predictions


def calculate_performance(network, x, y_true):
    predictions = network.predict_proba(x)
    y_pred = probas_to_classes(predictions)
    LOGGER.info(predictions.shape)
    LOGGER.info(y_pred.shape)
    LOGGER.info(y_true.shape)
    template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

    # Overall
    for labels_range, frequency, message in zip(margins, ['Overall', 'Frequent', 'Few', 'Zero'],
                                                ['Overall', 'Frequent Labels (>=10 Occurrences in train set)',
                                                 'Few-shot (<=10 Occurrences in train set)', 'Zero-shot (No Occurrences in train set)']):
        start, end = labels_range
        if start == end:
            continue
        LOGGER.info(message)
        LOGGER.info('----------------------------------------------------')
        for average_type in ['micro', 'macro', 'weighted']:
            p = precision_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            r = recall_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            f1 = f1_score(y_true[:, start:end], y_pred[:, start:end], average=average_type)
            LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

        for i in range(1, 11):
            r_k = mean_recall_k(y_true[:, start:end], predictions[:, start:end], k=i)
            p_k = mean_precision_k(y_true[:, start:end], predictions[:, start:end], k=i)
            rp_k = mean_rprecision_k(y_true[:, start:end], predictions[:, start:end], k=i)
            ndcg_k = mean_ndcg_score(y_true[:, start:end], predictions[:, start:end], k=i)
            LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
        LOGGER.info('----------------------------------------------------')


# SetUp Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nomothesia_nlp.common.text_preprocessor.spacy_tagger import Tagger
import unicodedata
import re
import xgboost as xgb

tagger = Tagger()


def word_tokenize_gr(tokens):
    norm_tokens = []
    for token in tokens.split():
        if re.search('[Α-ΩΆ-Ώα-ωά-ώ]', token, flags=re.IGNORECASE):
            norm_tokens.append(re.sub(r'\d', 'd', ''.join(
                (c for c in unicodedata.normalize('NFD', token.upper().strip(' ')) if unicodedata.category(c) != 'Mn')).strip(' ')))

    return norm_tokens


# Run Classifier
x_train, y_train = parse_dataset('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/train/*.json')
x_eval, y_eval = parse_dataset('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/dev/*.json')
x_val, y_val = parse_dataset('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/dev/*.json', one_hot=True)
x_test, y_test = parse_dataset('/home/chris/nomothesia.nlp.training/nomothesia_nlp/data/datasets/raptarchis/test/*.json', one_hot=True)

LOGGER.info(x_train.shape)
LOGGER.info(y_train.shape)

LOGGER.info(x_val.shape)
LOGGER.info(y_val.shape)

LOGGER.info(x_test.shape)
LOGGER.info(y_test.shape)
count = 1

vect = TfidfVectorizer(ngram_range=(1, 3), max_features=400000, stop_words=greek_stopwords,  lowercase=False, tokenizer=word_tokenize_gr)
vect = vect.fit(x_train)
svd = TruncatedSVD(algorithm='randomized', n_components=2000)
tfidf_train = vect.transform(x_train)
svd = svd.fit(tfidf_train)

x_train = svd.transform(tfidf_train)

tfidf_eval = vect.transform(x_eval)
x_eval = svd.transform(tfidf_eval)

tfidf_val = vect.transform(x_val)
x_val = svd.transform(tfidf_val)

tfidf_test = vect.transform(x_test)
x_test = svd.transform(tfidf_test)

for max_depth in [4, 5, 7, 10]:
    for n_estimators in [800]:
            for min_child_weight in [2, 5, 10]:
                LOGGER.info('=' * 50)
                LOGGER.info('Trial {}/12 SVD: {} MAX_DEPTH: {} ESTIMATORS: {} MIN_CHILD: {}'.format(count, 1000, max_depth, n_estimators, min_child_weight))
                LOGGER.info('=' * 50)

                text_clf = xgb.XGBClassifier(objective="multi:softmax", n_jobs=-1, max_depth=max_depth,
                                             n_estimators=n_estimators, booster='gbtree', min_child_weight=min_child_weight)

                text_clf.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], early_stopping_rounds=10, eval_metric='mlogloss')

                # Evaluation Report
                LOGGER.info('VALIDATION EVALUATION')
                LOGGER.info('=====================')

                calculate_performance(text_clf, x_val, y_val)

                LOGGER.info('TEST EVALUATION')
                LOGGER.info('=====================')
                calculate_performance(text_clf, x_test, y_test)
                count += 1
