import os

# specify the directory containing the files to be renamed
directory = 'imgTrain/citra_fertil/'

# loop through each file in the directory
for i, filename in enumerate(os.listdir(directory)):
    # create the new filename by appending the unique number to the old filename
    new_filename = f"fertil_{i+1}.jpg"
    
    # construct the full path for the old and new filenames
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_filename)
    
    # rename the file
    os.rename(old_path, new_path)
