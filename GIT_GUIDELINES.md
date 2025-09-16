
---
<div align="center">
   <strong style="font-size:1.2em;">"If you create merge hell, you fix merge hell."</strong>
</div>

<div align="center">
  
   ```mermaid
   flowchart LR
         A[Your feature branch] -- many changes --> B[Merge hell]
         B -- responsibility --> C[You fix it]
         style B fill:#ffcccc,stroke:#d33,stroke-width:2px
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

TODO: to be completed by Bryan.



