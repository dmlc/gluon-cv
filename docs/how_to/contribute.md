Contribute to GluonCV
=====================

GluonCV community is more than welcome to receive contributions from everyone!
Latest documentation can be found
[here](http://gluon-cv.mxnet.io/index.html).

There are lots of opportunities for you to become our
[contributors](https://github.com/dmlc/gluon-cv/graphs/contributors):

-   Ask or answer questions on [GitHub
    issues](https://github.com/dmlc/gluon-cv/issues).
-   Propose ideas, or review proposed design ideas on [GitHub
    issues](https://github.com/dmlc/gluon-cv/issues).
-   Improve the
    [documentation](http://gluon-cv.mxnet.io/index.html).
-   Contribute bug reports [GitHub
    issues](https://github.com/dmlc/gluon-cv/issues).
-   Write new
    [scripts](https://github.com/dmlc/gluon-cv/tree/master/scripts) to
    reproduce state-of-the-art results.
-   Write new
    [tutorials](https://github.com/dmlc/gluon-cv/tree/master/docs/tutorials).
-   Write new [public
    datasets](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/data)
    (license permitting).
-   Most importantly, if you have an idea of how to contribute, then do
    it!

For a list of open starter tasks, check [good first
issues](https://github.com/dmlc/gluon-cv/labels/good%20first%20issue).

How-to
------

-   [Make changes](#make-changes)
-   [Contribute scripts](#contribute-scripts)
-   [Contribute tutorials](#contribute-tutorials)
-   [Contribute new API](#contribute-new-api)
-   [Git Workflow Howtos](#git-workflow-howtos)

    > -   [How to submit pull request](#how-to-submit-pull-request)
    > -   [How to resolve conflict with master](#how-to-resolve-conflict-with-master)
    > -   [How to combine multiple commits into one](#how-to-combine-multiple-commits-into-one)
    > -   [What is the consequence of force push](#what-is-the-consequence-of-force-push)

### Make changes

Our package uses continuous integration and code coverage tools for
verifying pull requests. Before submitting, contributor should perform
the following checks:

-   [Lint (code style)
    check](https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L6-L11).
-   [Py2](https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L23-L35)
    and
    [Py3](https://github.com/dmlc/gluon-cv/blob/master/Jenkinsfile#L54-L66)
    tests.

### Contribute Scripts

The [scripts](https://github.com/dmlc/gluon-cv/tree/master/scripts) in
GluonCV are typically for reproducing state-of-the-art (SOTA) results,
or for a simple and interesting application. They are intended for
practitioners who are familiar with the libraries to tweak and hack. For
SOTA scripts, we usually request training scripts to be uploaded
[here](https://github.com/dmlc/web-data/tree/master/gluoncv/logs), and
then linked to in the example documentation.

See [existing
examples](https://github.com/dmlc/gluon-cv/tree/master/scripts).

### Contribute Tutorials

Our [tutorials](https://gluon-cv.mxnet.io/build/examples_classification/index.html) are
intended for people who are interested in CV and want to get better
familiarized on different parts in CV. In order for people to easily
understand the content, the code needs to be clean and readable,
accompanied by good quality writing.

See [existing
examples](https://gluon-cv.mxnet.io/build/examples_classification/index.html).

### Contribute new API

There are several different types of APIs, such as \*model definition
APIs, public dataset APIs, and building block APIs\*.

*Model definition APIs* facilitate the sharing of pre-trained models. If
you\'d like to contribute models with pre-trained weights, you can [open
an issue](https://github.com/dmlc/gluon-cv/issues/new) and ping
committers first, we will help with things such as hosting the model
weights while you propose the patch.

*Public dataset APIs* facilitate the sharing of public datasets. Like
model definition APIs, if you\'d like to contribute new public datasets,
you can [open an issue](https://github.com/dmlc/gluon-cv/issues/new)
and ping committers and review the dataset needs. If you\'re unsure,
feel free to open an issue anyway.

Finally, our *data and model building block APIs* come from repeated
patterns in examples. It has the highest quality bar and should always
starts from a good design. If you have an idea on proposing a new API,
we encourage you to [draft a design proposal
first](https://github.com/dmlc/gluon-cv/labels/enhancement), so that
the community can help iterate. Once the design is finalized, everyone
who are interested in making it happen can help by submitting patches.
For designs that require larger scopes, we can help set up GitHub
project to make it easier for others to join.

### Contribute Docs

Documentation is no less important than code. Good documentation
delivers the correct message clearly and concisely. If you see any issue
in the existing documentation, a patch to fix is most welcome! To locate
the code responsible for the doc, you may use \"View page source\" in
the top right corner, or the \"\[source\]\" links after each API. Also,
\"\[git grep\]\" works nicely if there\'s unique string.

### Git Workflow Howtos

#### How to submit pull request

-   Before submit, please rebase your code on the most recent version of
    master, you can do it by

``` {.sourceCode .bash}
git remote add upstream https://github.com/dmlc/gluon-cv
git fetch upstream
git rebase upstream/master
```

-   If you have multiple small commits, it might be good to merge them
    together(use git rebase then squash) into more meaningful groups.
-   Send the pull request!
    -   Fix the problems reported by automatic checks
    -   If you are contributing a new module or new function, add a
        test.

#### How to resolve conflict with master

-   First rebase to most recent master

``` {.sourceCode .bash}
# The first two steps can be skipped after you do it once.
git remote add upstream https://github.com/dmlc/gluon-cv
git fetch upstream
git rebase upstream/master
```

-   The git may show some conflicts it cannot merge, say
    `conflicted.py`.

    -   Manually modify the file to resolve the conflict.
    -   After you resolved the conflict, mark it as resolved by

    ``` {.sourceCode .bash}
    git add conflicted.py
    ```

-   Then you can continue rebase by

``` {.sourceCode .bash}
git rebase --continue
```

-   Finally push to your fork, you may need to force push here.

``` {.sourceCode .bash}
git push --force
```

#### How to combine multiple commits into one

Sometimes we want to combine multiple commits, especially when later
commits are only fixes to previous ones, to create a PR with set of
meaningful commits. You can do it by following steps. - Before doing so,
configure the default editor of git if you haven't done so before.

``` {.sourceCode .bash}
git config core.editor the-editor-you-like
```

-   Assume we want to merge last 3 commits, type the following commands

``` {.sourceCode .bash}
git rebase -i HEAD~3
```

-   It will pop up an text editor. Set the first commit as `pick`, and
    change later ones to `squash`.
-   After you saved the file, it will pop up another text editor to ask
    you modify the combined commit message.
-   Push the changes to your fork, you need to force push.

``` {.sourceCode .bash}
git push --force
```

#### Reset to the most recent master

You can always use git reset to reset your version to the most recent
master. Note that all your **\*local changes will get lost**\*. So only
do it when you do not have local changes or when your pull request just
get merged.

``` {.sourceCode .bash}
git reset --hard [hash tag of master]
git push --force
```

#### What is the consequence of force push

The previous two tips requires force push, this is because we altered
the path of the commits. It is fine to force push to your own fork, as
long as the commits changed are only yours.
