import sys
import os
from threading import Thread
from skimage import io

# stderr print
def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

def mkdir(path):
    import os
    # remove front blanks
    path = path.strip()
    # remove / and \ at the end
    path = path.rstrip('/')
    path = path.rstrip('\\')

    if os.path.exists(path):
        return False
    else:
        os.makedirs(path)
        return True

def move_to(file, path):
    if not os.path.isfile(f):
        eprint('Could not find file {}'.format(f))
        return None
    f_split = os.path.split(file)
    new_f = os.path.join(path, f_split[1])
    eprint('Moving...\n  {}\n  {}'.format(file, new_f))
    try:
        os.rename(file, new_f)
    except Exception as err:
        eprint(err)
    return new_f

# recursively list all the files' path under directory
# rename file names not supported by current encoding
def listdir_files(path, recursive=True, filter_ext=None, encoding=None):
    import os, locale
    if encoding is True: encoding = locale.getpreferredencoding()
    if filter_ext is not None: filter_ext = [e.lower() for e in filter_ext]
    files = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        for f in file_names:
            if os.path.splitext(f)[1].lower() in filter_ext:
                file_path = os.path.join(dir_path, f)
                try:
                    if encoding: file_path = file_path.encode(encoding)
                    files.append(file_path)
                except UnicodeEncodeError as err:
                    eprint(file_path)
                    new_file = file_path.encode(encoding, 'xmlcharrefreplace')
                    eprint(new_file)
                    os.rename(file_path, new_file.decode(encoding))
                    files.append(new_file)
        if not recursive: break
    return files

# constants
NUM_THREADS = 8
DATASET_PATH = r'E:\MyTF\Dataset.SR\Train\Konachan'
CORRUPT_PATH = r'E:\Crawler\Konachan\Corrupt'
UNEXPECTED_PATH = r'E:\Crawler\Konachan\Unexpected'
LARGE_PATH = r'E:\Crawler\Konachan\Large'
mkdir(CORRUPT_PATH)
mkdir(UNEXPECTED_PATH)
mkdir(LARGE_PATH)

# rename Unicode names
if False:
    files = listdir_files(DATASET_PATH, filter_ext=['.jpeg', '.jpg', '.png'],
                          encoding=True)
    print('Total files: {}'.format(len(files)))

# filter corrupt files
def parseFiles(files, start=0, end=None):
    if end is None: end = len(files)
    for _ in range(start, end):
        f = files[_]
        if _ % 1000 == 0: eprint(_)
        try:
            img = io.imread(f)
        except Exception as err:
            eprint(err)
            #new_path = os.path.join(f_split[0], 'corrupt')
            #mkdir(new_path)
            #new_f = os.path.join(new_path, f_split[1])
            move_to(f, CORRUPT_PATH)

# filter corrupt files
if False:
    # get file list
    files = listdir_files(DATASET_PATH, filter_ext=['.jpeg', '.jpg', '.png'],
                          encoding=None)
    
    index = 0
    slice = (len(files) + NUM_THREADS) // NUM_THREADS
    threads = []

    for _ in range(NUM_THREADS):
        start = index
        index = min(len(files), index + slice)
        args = (files, start, index)
        threads.append(Thread(target=parseFiles, args=args, daemon=False))
        threads[_].start()

# filter unexpected files
if False:
    # get file list
    files = listdir_files(DATASET_PATH, filter_ext=['.jpeg', '.jpg', '.png'],
                          encoding=None)
    
    for f in files:
        fl = f.lower()
        if fl.find('waifu2x') >= 0 or fl.find('artifact') >= 0 or fl.find('.gif') >= 0:
            move_to(f, UNEXPECTED_PATH)

# filter large files
if False:
    # get file list
    files = listdir_files(DATASET_PATH, filter_ext=['.jpeg', '.jpg', '.png'],
                          encoding=None)
    
    for f in files:
        size = os.stat(f).st_size
        ext = os.path.splitext(f)[1].lower()
        if (ext == '.jpg' or ext == '.jpeg'):
            if size < (3 << 20): continue
        elif size < (4 << 20):
            continue
        eprint('{} Bytes ({})'.format(size, ext))
        move_to(f, LARGE_PATH)


