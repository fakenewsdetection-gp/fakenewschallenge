from csv import DictReader


class Dataset:
    """
    Define class for tha dataset.
    """
    def __init__(self, instances_filepath, bodies_filepath):
        # Load data
        self.instances = self.read(instances_filepath)
        bodies = self.read(bodies_filepath)
        self.heads = {}
        self.bodies = {}
        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])
        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filepath):
        """
        Read data from CSV file.

        Args:
            filepath: str, representing the path of the csv file.

        Returns:
            data: list, representing data instances in the form of dictionaries.
        """
        data = []
        with open(filepath, "r", encoding='utf-8') as file:
            dict_reader = DictReader(file)
            for line in dict_reader:
                data.append(line)
        return data
