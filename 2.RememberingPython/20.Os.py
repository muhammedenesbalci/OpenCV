print("# OS------------------------------")
import os

# Our os system
print("Ours os------- nt = widnows")
print(os.name)

# Current dir
print("Current dir--------------------------")
current_dir = os.getcwd()
print(current_dir)

# Create folder
folder_name = "new_folder"
# os.mkdir(folder_name)

# Rename Folder
new_folder_name = "new_folder2"
# os.rename(folder_name, new_folder_name)

# Go to folder
os.chdir(current_dir + "\\" + new_folder_name)
print(os.getcwd())

# Looking files in a file
os.chdir(current_dir)
files = os.listdir()
print(files)

# Choose by file extension
for i in files:
    if i.endswith(".py"):
        print(i)

# Delete a file
os.rmdir(new_folder_name)

# does file exist
os.path.exists("Lesson_Codes")