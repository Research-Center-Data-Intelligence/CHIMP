
---
<div align="center">
   <strong style="font-size:1.2em;">"If you create merge hell, you fix merge hell."</strong>
</div>

<div align="center">
  
   ```mermaid
   flowchart LR
         A[Your feature branch] -- many changes --> B[Merge hell]
         B -- responsibility --> C[You fix it]
         style B fill:#ffcccc,stroke:#d33,stroke-width:2px,color:#000
         style C fill:#ccffcc,stroke:#393,stroke-width:2px
   ```

</div>

---

(so here are some guidelines...)

## Branching Strategy

For this project, we use a branching strategy that fits open source development:

- **Main branch:** The `main` branch is the base branch of the repository. It always contains the most stable and up-to-date version of the code.
- **Feature branches:** Staff members work on new features, bug fixes, or improvements in their own feature branch. These branches are created from the `main` branch and, once finished, are merged back via a pull request.
- **Forks for students:** Students create a fork of the repository in their own GitHub environment. They work in their own fork and can contribute to the main project via pull requests.

This approach ensures a clear separation between stable code, active development by staff, and external contributions by students.

## Avoid long-lived branches

It is important to bring changes from a feature branch or fork back to the `main` branch in a timely manner via a pull request. Long-lived branches with many changes that do not reach `main` increase the risk of merge conflicts and make it harder to integrate new functionality. By regularly creating a pull request and merging your work with `main`:

- The codebase remains organized and up-to-date.
- You prevent your work from becoming outdated or difficult to integrate.
- You contribute more quickly to the overall project result.

In short: keep branches short and merge them in time to maintain a healthy and collaborative development environment.

## Regularly sync with main

It is essential to regularly synchronize your feature branch or fork with the `main` branch. This prevents you from falling behind on recent changes, bug fixes, or new features added by others. By syncing frequently:

- You minimize merge conflicts when merging your work.
- You ensure you are developing based on the most current and stable code.
- You can respond more quickly to changes in the project.

## How to sync with main?
Below are the command line options; it is also possible to do this in the VSCode GUI.

### In a feature branch (same repository)
1. Make sure your local repository is up-to-date:
   ```
   git fetch origin
   ```
2. Checkout your feature branch:
   ```
   git checkout <your-feature-branch>
   ```
3. Merge the latest changes from main:
   ```
   git merge origin/main
   ```
4. Resolve any merge conflicts, commit, and push your branch if needed.

### In a fork (own repository)
1. Add the original project as a remote (if not already done):
   ```
   git remote add upstream https://github.com/Research-Center-Data-Intelligence/CHIMP.git
   ```
2. Fetch the latest changes from the main project:
   ```
   git fetch upstream
   ```
3. Checkout your own branch:
   ```
   git checkout <your-branch>
   ```
4. Merge the changes from main in the main project:
   ```
   git merge upstream/main
   ```
5. Resolve any merge conflicts, commit, and push your branch if needed.



## Pull request guidelines

### Main branch

TODO: to be completed by Bryan.

In order to preserve linear history on the `main` branch, pull requests to main are enforced to be

!! Do not use merge commits for feature branches !!


#### Proposal 1
> **1.1 Rebase (Guided)**
> The guided rebase proposal for pull requests is a method of using the terminal to rebase feature branches onto the main branch in a safe and approachable way such that individual commits are retained in commit history while ensuring a linear history.
> 
> <span style="text-decoration:underline">Approach</span>
> 1. Configure your local git settings for the repository like such:
> ```bash
> # Rebase config
> git config pull.rebase true
> git config rebase.autostash true
> # (Optional) Conflict resolve
> git config rerere.enabled true
> ```
> **<div align="center">or (for all repositories)</div>**
> ```bash
> # Rebase config
> git config --global pull.rebase true
> git config --global rebase.autostash true
> # (Optional) Conflict resolve
> git config --global rerere.enabled true
> ```
> 2. Create/check out branch:
> ```bash
> git checkout -b feature/name main
> ```
> **<div align="center">or</div>**
>
> ```bash
> git checkout
> ```
> 3. Rebase branch onto latest main
> ```bash
> git fetch origin
> git rebase origin/main
> ```
> **<div align="center">or</div>**
> ```bash
> git fetch upstream
> git rebase upstream/main
> ```
> 4. Fix any merge conflicts. For each commit ahead of main, in chronological order conflicts will be presented, which can be fixed as you would with any other merge conflict.
> 5. Push the local rebase (force with lease)
> ```bash
> git push --force-with-lease
> ```
> 6. Open a PR on github.
>
> _Exception_
> If the commit history is a mess already, the operations staff [ref to contacts] should be contacted. During exceptional circumstances the linear history enforcement can be temporarily disabled.
> 
> <span style="text-decoration:underline">Pros</span>
> 1. Linear history keeps track of individual changes
> 2. Since local configuration needs to be updated, we can introduce `git config rerere.enabled true` which adds conflict memory which automatically solves repeated merge conflict resolutions more easily (though this might also become a crutch and a burden).
> 3. Teaches more git complex principals to students
> 
> <span style="text-decoration:underline">Cons</span>
> A rebase can cause issues if you're unaware how merge.
> 1. For syncing with main, the final sync should be a rebase such that any rebase conflicts have been dealt with.
> 2. For pull request configuring any local pulls to be rebase is required to prevent merge accidental merge conflicts between the local and remote after a rebase. This prevents duplicating commits due to a sync as rebase commit hashes are different from the original commit hashes.
> 3. Rebase may complain about uncommited changes. To not have to struggle with understanding stashes (though I'd recommend knowing how to work with git stashes in safe fashion), another git configuration needs to be set: `git config rebase.autostash true`. This automatically stashes all uncommitted changes until you've completed the rebase and then reintroduces the changes again.
> 4. Requires a force push after rebase, as the rebase means the hashes of commits don't match (see con 2). However, by using `git push --force-with-lease` you can make sure you only force if the remote branch doesn't have any changes to be pulled.
>
> **1.2 Rebase (Scripted)**
>
> Unlike the guided rebase, configuration and (optionally) the rebase will be scripted with shell/bash for Linux or bat/ps for Windows. MacOS... Send a support ticket to include that one, as we currently don't have any hardware to test the scripts on.
> <span style="text-decoration:underline">Approach</span>
> 
> 1. Execute the `initialise_repo` script
> 2. Same as guide _or_ execute the following `rebase` script.
> 
> <span style="text-decoration:underline">Pros</span>
> 1. The same as the guided approach
> 2. Less error prone
> 3. Rerere might not be a safe configuration, since it 
> 
> <span style="text-decoration:underline">Cons</span>
> 1. Magic box solves everything, so magic problems become difficult to understand and solve
> 2. The scripts could be ignored (but if you cause merge hell, you fix merge hell, in which case see con 1)
> 3. Overrides original feature commits such that a merge commited pre-rebase could duplicate commits.

**Proposal 2**
> <span style="text-decoration:underline">Squash Merge</span>
>
> Squash merge, unlike rebase, adds a single commit to the `main` such that all changes introduced in the feature branch are still linearly added into main.
>
>
> <span style="text-decoration:underline">Approach</span>
> 1. Create/check out branch:
> ```bash
> git checkout -b feature/name main
> ```
> **<div align="center">or</div>**
>
> ```bash
> git checkout
> ```
> 2. Merge lastest `main` into feature branch
> ```bash
> git fetch origin
> git merge origin/main
> ```
> **<div align="center">or</div>**
> ```bash
> git fetch upstream
> git merge upstream/main
> ```
> 3. Fix any merge conflicts and commit the resolutions if need as you would submit a regular commit.
> ```bash
> git commit -m 'message'
> git pull
> git push
> ```
> 6. Open a PR on github.
>
> <span style="text-decoration:underline">Pros</span>
> 1. Less error prone, as squash merge still inherently behaves like a standard merge whilst keeping the `main` commit history linear.
> 2. More emphasis can be put on make feature smaller and more concise, which is a good practice to learn and understand.
> 
> <span style="text-decoration:underline">Cons</span>
> 1. This would require dilligence, as feature branches need to be more attomic. Large multi-file changes would counter the otherwise more readable linear history.
> 2. The only way to rollback changes on `main` is by undoing entire feature branches, or appending a change to undo changes instead. This means less history (and thus version) control.

**Proposal 3**
> <span style="text-decoration:underline">Allow basic merge</span>
>
> Forgo linear merging, and use the standard merge to pull changes from feature branches into the main branch.
>
> <span style="text-decoration:underline">Approach</span>
> 1. Create/check out branch:
> ```bash
> git checkout -b feature/name main
> ```
> **<div align="center">or</div>**
>
> ```bash
> git checkout
> ```
> 2. Merge lastest `main` into feature branch
> ```bash
> git fetch origin
> git merge origin/main
> ```
> **<div align="center">or</div>**
> ```bash
> git fetch upstream
> git merge upstream/main
> ```
> 3. Fix any merge conflicts and commit the resolutions if need as you would submit a regular commit.
> ```bash
> git commit -m 'message'
> git pull
> git push
> ```
> 6. Open a PR on github.
>
> <span style="text-decoration:underline">Pros</span>
> 1. Less error prone, as if the main has been synced regularly no conflicts can remain, nor can they be introduced that easily.
> 2. Should still be safe as not too many people are working on different branches simultaneously.
> 
> <span style="text-decoration:underline">Cons</span>
> 1. Readability of the commit history will become very illegible if lots of people work on features simultaneously
> 2. Rollbacks maybe become more difficult to do when the commit history is illegible.

### Feature branch

!! Merging is also allowed, however PR's from one feature branch into another should only happen if the source branch cannot be pulled into main due to stability issue. In which case do ask yourself if it's a good idea to merge from that branch into the target branch at all !!
