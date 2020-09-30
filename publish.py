""" Push data update as commit, to be used in artifacts
"""

from git import Repo, exc
import json
import datetime



PATH_OF_GIT_REPO = r'.git'  # make sure .git folder is properly configured

global_file_name = "docs/_data/global_info.json"
files_to_add = [ #todo: clean paths
    global_file_name,
    "docs/_data/images_info.json",
]


def get_today_date_str(): #todo: use in main
    return datetime.date.today().strftime("%B %d, %Y")

def get_date_last_update():
    """ Date last update according to json info"""
    with open(global_file_name) as json_file:
        data = json.load(json_file)
        return data["date_last_update"]
    return None

# check_today
def is_outdated(former_date):
    if former_date is None:
        return True
    return get_date_last_update() != former_date

def git_push(is_dummy, do_push):
    """ Push a commit for modifications of selected docs/ files
            is_dummy: dummy commit name for test purpose
            do_push: avoid real push, just simulate it
    """
    branch_name = "dummy_branch" if is_dummy else "master"
    repo = Repo(PATH_OF_GIT_REPO)
    try:
        repo.git.checkout('HEAD', b=branch_name) #todo: just switch, don't create...
    except exc.GitCommandError as e:
        print("fails to checkout:", e)
        pass
    repo.git.add(files_to_add)

    commit_message = 'Publish: Example commit' if is_dummy \
        else "Data: daily update (automatic commit)"
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    print("Committed {}".format(commit_message))
    if do_push:
        origin.push()
        print("Pushed {}".format(commit_message))

def push_if_outdated(do_push, former_date):
    # pushing even of not outdated, because github bots won't run the action 
    # if is_outdated(former_date):
    git_push(False, do_push)
    # else:
    #    print("Not pushing (not outdated)")

if __name__ == '__main__':
    print("is outdated:", is_outdated())
    git_push(True, False)
