from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


class NIH_Google_SciRep_Dataset(Dataset):
    """A relabelling of a subset of images from the NIH dataset.  The data tables should
    be applied against an NIH download.  A test and validation split are provided in the
    original.  They are combined here, but one or the other can be used by providing
    the original csv to the csvpath argument.

    Chest Radiograph Interpretation with Deep Learning Models: Assessment with
    Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
    Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J. Reicher, Scott Mayer
    McKinney, Gavin E. Duggan, Krish Eswaran, Po-Hsuan Cameron Chen, Yun Liu,
    Sreenivasa Raju Kalidindi, Alexander Ding, Greg S. Corrado, Daniel Tse, and
    Shravya Shetty. Radiology 2020

    https://pubs.rsna.org/doi/10.1148/radiol.2019191293

    NIH data can be downloaded here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=os.path.join(datapath, "google2019_nih-chest-xray-labels.csv.gz"),
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(NIH_Google_SciRep_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Other"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv[pathology] == "YES"

            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # rename pathologies
        self.pathologies = list(self.pathologies)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample
