import torch


class Dataset:
    def __init__(self, data_path, cutoff):
        self.data_path = data_path
        self.cutoff = cutoff
        print('cutoff:', self.cutoff)
        if self.cutoff not in [3, 5, 10, 20]:
            raise ValueError('Unknown cutoff value')

    def get_data(self):
        raw_dataset = torch.load(self.data_path)
        ls = []
        for data in raw_dataset:
            # change edge_index according to cutoff
            if self.cutoff == 3:
                data.edge_index = data.edge_index_3
            elif self.cutoff == 5:
                data.edge_index = data.edge_index_5
            elif self.cutoff == 10:
                data.edge_index = data.edge_index_10
            elif self.cutoff == 20:
                data.edge_index = data.edge_index_20

            # exclude single atom
            if data.edge_index.size()[0] == 0:
                print('exclude: ', data.name)
                pass
            else:
                ls.append(data)

        return ls

