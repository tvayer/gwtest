
def depth_iter(element, tag=None):
    stack = []
    stack.append(iter([element]))
    nodenumber=0
    while stack:
        e = next(stack[-1], None)
        if e == None:
            stack.pop()
        else:
            stack.append(iter(e))
            nodenumber+=1
            if tag == None or e.tag == tag:
                yield (nodenumber,e, stack.__len__()-1) 

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract_xml_attribute(string, start='[', stop=']'):
    return string[string.index(start)+1:string.index(stop)]


def extract_attribute(element):
    attribute = extract_xml_attribute(element.attrib.get('Attribute')).split(" ")
    return [float(i) for i in attribute[1:len(attribute)]]

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def dict_argmax(d):
    return max(d, key=d.get)

def dict_argmin(d):
    return min(d, key=d.get)

def read_files(mypath):
    from os import listdir
    from os.path import isfile, join

    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


