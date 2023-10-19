


import numpy as np
import pandas

# from ..util.io import NTuple


class DataProxy:
    """ Provide flexible ntuple manipulation.

    In particular allows to map fields of a ntuple to arbitrary field
    names and eases the creation of index for fields.
    """
    def __init__(self, datasrc, **keys):
        """Parameters:
	-----------

	keys: provides a mapping between fields of the ntuple and
	arbitrary names. For exemple, keys mag='mmag' will create
	the view: dp.mag = nt['mmag']
	"""
	# Load data
        self.nt = datasrc
        self.mapping = keys
        self._create_views()
        self._index_list=set()
        self._proxy_list=[]
        self._external_field_list=[]

    def __len__(self):
        return len(self.nt)
        
    def _create_views(self):
        # Create views
        for k, v in self.mapping.items():
            setattr(self, k, self.nt[v])

    def add_views(self, **keys):
        """ A view is a mapping of a NTuple field: dp.key = nt['otherkeyname']"""
        for k, v in keys.items():
            setattr(self, k, self.nt[v])
        self.mapping.update(keys)

    def _update_proxys(self):
        for k, v in self._proxy_list:
            setattr(self,k,v(self))

    def add_proxy(self, name, f):
        self._proxy_list.append((name, f))
        setattr(self,name,f(self))

    def add_partial(self, name, data, index):
        #self._partial_list.append((name,data,index))
        s = np.zeros(index.max()+1)
        s[index.astype('int')] = data
        setattr(self, name, s)

    def add_field(self, name, data):
        """ Add external data sharing the same number of records.

	There is no mem copy so this is more memory efficient than
	just catenating the field to the existing NTuple."""
        if name not in self._external_field_list:
            self._external_field_list.append(name)
        if isinstance(self.nt, (pandas.DataFrame, pandas.Series)):
            setattr(self, name, pandas.Series(data))
        else:
            setattr(self, name, data)
            
    def _update_external_fields(self, condition):
        for name in self._external_field_list:
            setattr(self, name, getattr(self,name)[condition])

    def compress(self, condition):
        self.nt = self.nt[condition].squeeze()
        self._create_views()
        self._update_external_fields(condition)
        self._update_proxys()
        self._update_index(condition)

    def make_index(self, fieldname, intmap=False):
        """ Build an index of the values taken by field fieldname.
        
        Exemples:
        ---------
        Let assume that self.mjd = [12,14,14]
        calling self.make_index('mjd') will create:
        self.mjd_set = [12,14]
        self.mjd_map = {12:1, 14:2}
        self.mjd_index = [1,2,2]
        """
        self.update_index(fieldname, intmap)
        self._index_list.add((fieldname, intmap))

    def update_index(self, fieldname, intmap=False):
        if intmap:
            a = getattr(self,fieldname).astype('int')
            s = set(a)
            s = list(s)
            s.sort()
            miv = a.min()
            mav = a.max()
            m = np.ones(mav-miv+1, dtype='int') * -1
            s = np.fromiter(s,dtype='int')
            for i, e in enumerate(s):
                m[e-miv] = i
            setattr(self, fieldname+'_set',s)
            setattr(self, fieldname+'_map',m)
            setattr(self, fieldname+'_index', m[a - miv])
        else:
            a = getattr(self,fieldname)
            s = set(a)
            s = np.array(list(s)) #np.fromiter(s, dtype=type(a[0]))
            m = dict(list(zip(s, list(range(len(s))))))
            setattr(self, fieldname+'_set',s)
            setattr(self, fieldname+'_map', m)
            i = np.fromiter((m[e] for e in a),'int')
            setattr(self, fieldname+'_index', i)

    def _update_index(self,condition):
        for fieldname, intmap in self._index_list:
            self.update_index(fieldname, intmap)
	#for ind in (getattr(self, i) for i in self._index_list):
	#    ind = ind[condition]
