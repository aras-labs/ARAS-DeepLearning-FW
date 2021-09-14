import os
import shutil


def delete_empty_logs():
    for root, d_names, f_names in os.walk('./logs'):
        if os.path.basename(root) == 'weights':
            if len(f_names) <= 1:
                shutil.rmtree(root)
                print(f_names)

    for root, d_names, f_names in os.walk('./logs'):
        if os.path.basename(root) != 'weights':
            if len(f_names) > 1 and not ('weights' in d_names):
                shutil.rmtree(root)
                print(root)

    for root, d_names, f_names in os.walk('./logs', topdown=False):
        if len(d_names) == 0 and len(f_names) == 0:
            shutil.rmtree(root)


if __name__ == "__main__":
    print('hi')
    delete_empty_logs()
