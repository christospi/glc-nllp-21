class DataNode(object):
    def __init__(self, type='', id='', title='', text=''):
        self.type = type
        self.id = id
        self.title = title
        self.text = text
        self.year = None
        self.law_id = None
        self.leg_uri = None
        self.children = []

    def print_tree(self):
        if self.type == '':
            pass
        elif self.type == 'ΤΟΜΟΣ':
            print(self.type, self.id, self.title)
        elif self.type == 'ΚΕΦΑΛΑΙΟ':
            print('\t' + self.type, self.id, self.title)
        elif self.type == 'ΘΕΜΑ':
            print('\t\t' + self.type, self.id, self.title)
        elif self.type == 'ΑΡΘΡΟ':
            print('\t\t\t\t' + self.type, self.id)
        else:
            print('\t\t\t' + self.type, self.id)
        for dn in self.children:
            dn.print_tree()

    def sort_tree(self):
        if self.children:
            for sn in self.children:
                sn.sort_tree()
            self.children = sorted(self.children, key=lambda DataNode: DataNode.id)

    def search_vol_by_label(self, label):
        for vol in self.children:
            if vol.title == label:
                return vol
        return None
