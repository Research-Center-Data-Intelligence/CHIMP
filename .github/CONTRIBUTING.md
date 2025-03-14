If you'd like to help improve CHIMP there are several ways to add value to the project:
- [Bugs](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#bugs) and [security](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#security-issues) issue reporting
- [Feature suggestions](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions)
- [Documentation maintenance](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#documentation)
- [Code submissions](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#submitting-changes)
- Use [CHIMP as a base](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#writing-plugins) for your own AI Ops project

[Short collaboration and project management summary]
- project type (open-source), application type, programming languages framework
- contributions, documentation, bug reports, suggestions, no design improvements, new plugins are welcome.
- contribution process described below for bug reports, suggestions and improvements.
- contribution standards described below for coding conventions (ESLint, PEP8, Prettier, ...; unit testing, documentation needs, PRs), commit messages conventions, PR update conventions
- Community discussion: GitHub Discussions, any other places are pending. Code of conduct still according to basic GitHub standard (small community type)
- Licensing: open-source; CLA will might be drafted to ensure commited code is legible to be made open-source and us as project maintainers are granted license to use modify and distribute contributions. Mostly to clarify the projects relationship to Research Center Data Intelligence. -CLA.md will follow if required, possibly a CLA assistant too somewhen in the future-


## Reporting issues
There are 3 types of issues you can submit: Bugs, suggestions and security issues.

### Duplicate issues
All issues can be submitted by simply opening a new issue. Before opening a new issue first try and make sure a similar issue doesn't exist yet. If such an issue does exist, do add a comment to the issue with the same information you'd provide if you were to open a new issue of the same type with the header **[duplicate]**. This helps us evaluate the impact of an issue better.

### Bugs
When submiting a bug there are few details you can provide as defined in the [bug report issue template](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/ISSUE_TEMPLATE/bug_report.md):
1. Bug description
2. Reproduction steps
3. Behaviour you expected
4. Screenshots (if applicable)
5. Device information (if you know)
Try to at least provide the first 3 of these details for the issue to be of proper value.

### Security issues
For now, security issues can be submitted as bug reports. Security issues are only reserved to the core [services](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main?tab=readme-ov-file#components) as definined in the readme file of this repository. Security issues limited to plugins and custom applications built atop CHIMP out of scope for project maintance. Would you like the latter issues to be solved, you are welcome to [submit your own fix](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#bugs-1) for the issue!

### Suggestions and questions
[Summary about how suggestions and questions are handled]
Would you have suggestions or questions for the project, feel free to open a ticket. Before submiting a question, do check the [FAQ](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main?tab=readme-ov-file#FAQ) in the project's readme.

To submit a [suggestion](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/ISSUE_TEMPLATE/feature_request.md) please provide:
1. Why you'd think a feature is useful based on either
    - A problem statement
    - Proposed added value
2. Feature description
3. Considered alternative solutions for functionality

To submit a [question](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/ISSUE_TEMPLATE/question-proposal.md) please provide:
1. Origin of question
2. Question statement

## Submitting changes
If you'd like to help improve CHIMP, there three different ways to have your own improvements consolidated. Accepted improvements include [solving bugs](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#bugs-1) or [adding features](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#features) pertaining to the core services supporting continual learning (non-plugins), as well as helping us keep our [documentation](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#documentation) up-to-date. If you're [developing your own plugin application](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#submitting-your-plugin) instead, no guarantee will be given to consider its contributions to the project.

### Repository Updates
Before submitting changes related to a bug, create an issue for the [bug](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#bugs) or [feature](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions) according to their respective guidelines. Create a pull request for the changes you've made and link the pull request in the issue. This helps make it easier to keep track of requests and process them. 

### Documentation
Feel free to not only update informative files within the repository, but also [suggest](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions) or write documentation that clarifies things you think should need (further) explanation. This may well be the most valuable way to help the project succeed.

## Coding conventions
**[TBD]**
Coding conventions will be further defined soon. Please check in next time you fork this project and before you request a pull request in case any such decisions have been made for this project/repository.

## Writing plugins
Generally, your custom [plugins applications](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main?tab=readme-ov-file#training-plugin-development) aren't always added to this repository but we do encourage writing your own plugin application atop of CHIMP. Even if not included in this repository directly, interesting plugins built with CHIMP may be included in the README as a shout-out to a valuable application/example of continual learning using CHIMP.

Would you need advise during the process of making the plugin application, you can always submit a [question issue](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions). We'll do our best to help you on your way if your issue isn't already properly documented. This way you help us make CHIMP more accessible for others as well.

### Submitting your plugin
When writing your own [plugins applications](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main?tab=readme-ov-file#training-plugin-development) there is no guarantee the plugin will be included as an official plugin in this repository. DO NOT write plugins with the mere intend of having it included in this repository without consulting the repository owner or maintainers first. You can do this by submitting a [suggestion issue]([https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main?tab=readme-ov-file#training-plugin-development](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions) so we can reach out to you.

Plugin applications in this repository are aimed at demonstrating the use of CHIMP's continual learning framework. If your plugin application does not support that goal, it is likely that your contribution will not be accepted into this repository. HOWEVER, applications you've made that beyond the size of a small demonstrator _can_ still receive a shout-out in the README of this repository instead. Don't hesitate to share!

If your plugin application suggestion is accepted, please fork this repository and develop your plugin application completely within your own repository first. Only finished plugin application submitted via pull request will be accepted. When submitting a pull request it can help us approve it faster if you reopen the suggestion issue and link the pull request in a comment. Would you need advise during the process of making the plugin application, you can either reopen the suggestion issue you made and write your question in as a comment or submit a [question issue](https://github.com/Research-Center-Data-Intelligence/CHIMP/tree/main/.github/CONTRIBUTING.md#suggestions-and-questions).
