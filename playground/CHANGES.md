git add \*
git commit
git push origin dev

git fetch upstream
git checkout main
git merge upstream/main
git push origin main

git checkout dev
git rebase main
