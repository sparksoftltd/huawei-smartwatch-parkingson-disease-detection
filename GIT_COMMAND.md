### Prerequisites
   ```sh
   git config --global user.email "you@example.com"
   git config --global user.name "Your Name"
   ```

#### Install git LFS (Large File Storage)
https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage


#### Configuring Git Large File Storage
https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage

#### Checkout a new branch and work on your code, this command only is required to run once, once it is checked out you are working on your own copy of the code.
   ```sh
   git checkout -b YOUR_BRANCH_NAME main
   ```


#### Add and commit your changes with message, it is recommended to commit often with context
  ```sh
  git commit -am "Add the feature extraction logic"
  ```


#### Push your changes to remote repo in github, if you dont push, your changes only exist in your local PC, and if your code is #deleted, you will lose it forever, hence it is recommended to push often
  ```sh
  git push origin YOUR_BRANCH_NAME
  ```


#### Rebase your changes with remote repo, this is required because when you are working on your code in your branch, someone #else may push their changes to remote repo, and to fetch their pushed changes, you can use this command to rebase their #changes into the branch you are working on currently
  ```sh
  git checkout main
  git pull --ff-only
  git checkout YOUR_BRANCH_NAME
  git rebase origin/main
  git push origin YOUR_BRANCH_NAME -f
  ```

