.. _governance:

========================================
Geomstats governance and decision-making
========================================

This document formalizes the governance process used by the
geomstats project, to clarify how decisions are made and how
elements of our community interact.

This document establishes a decision-making structure that takes into account
feedback from all members of the community and strives to find consensus, while
avoiding any deadlocks.

Geomstats is a meritocratic, consensus-based community project. Anyone with an
interest in the project can join the community, contribute to the project
design and participate in the decision making process. This document describes
how that participation takes place and how to earn merit within
the project community.

This document is inspired by Scikit-Learn's and Pandas' governance documents.

Roles And Responsibilities
==========================

Contributors
------------

Contributors are community members who contribute in concrete ways to the
project. Anyone can become a contributor, and contributions can take many forms
- not only code - as detailed in the `contributors guide <https://geomstats.github.io/contributing/index.html#contributing>`_.

Contributor Experience Team
---------------------------

The contributor experience team is composed of community members who have permission on
github to label and close issues. Their work is
crucial to improve the communication in the project and limit the crowding
of the issue tracker.

Similarly to what has been decided in the `python project
<https://devguide.python.org/triaging/#becoming-a-member-of-the-python-triage-team>`_,
any contributor may become a member of the geomstats contributor experience team,
after showing some continuity in participating to geomstats
development (with pull requests, reviews and help on issues).
Any core developer or member of the contributor experience team is welcome to propose a
geomstats contributor to join the contributor experience team. Other core developers
are then consulted: while it is expected that most acceptances will be
unanimous, a two-thirds majority is enough.
Every new member of the contributor experience team will be announced in Geomstats GoogleGroups mailing
list. Members of the team are welcome to participate in `monthly core developer meetings
<https://github.com/geomstats/admin/blob/main/meeting_notes.md>`_.

The contributor experience team is currently constituted of:

  * Luis Pereira,
  * Nina Miolane.

.. _communication_team:

Communication team
-------------------

Members of the communication team help with outreach and communication
for Geomstats. The goal of the team is to develop public awareness of
Geomstats, of its features and usage, as well as branding.

For this, they can operate the Geomstats accounts on various social
networks and produce materials.

Every new communicator will be announced in Geomstats GoogleGroups mailing list.
Communicators are welcome to participate in `monthly core developer meetings
<https://github.com/geomstats/admin/blob/main/meeting_notes.md>`_.

The communication team is currently constituted of:

  * Nina Miolane.

Core developers
---------------

Core developers are community members who have shown that they are dedicated to
the continued development of the project through ongoing engagement with the
community. They have shown they can be trusted to maintain Geomstats with
care. Being a core developer allows contributors to more easily carry on
with their project related activities by giving them direct access to the
project's repository.

Core developers are expected to review code
contributions, can merge approved pull requests, can cast votes for and against
merging a pull-request, and can be involved in deciding major changes to the
API.

New core developers can be nominated by any existing core developers. Once they
have been nominated, there will be a vote by the current core developers.
Voting on new core developers is one of the few activities that takes place on
the project's private management list. While it is expected that most votes
will be unanimous, a two-thirds majority of the cast votes is enough. The vote
needs to be open for at least 1 week.

Core developers that have not contributed to the project (commits or GitHub
comments) in the past 12 months will be asked if they want to become emeritus
core developers and recant their commit and voting rights until they become
active again. The list of core developers, active and emeritus (with dates at
which they became active) will be public on the Geomstats website.

The core developers are currently:

  * Luis Pereira, 
  * Nicolas Guigui, 
  * Alice Le Brigant, 
  * Jules Deschamps, 
  * Saiteja Utpala, 
  * Adele Myers, 
  * Anna Calissano,
  * Yann Thanwerdas,
  * Elodie Maignant,
  * Tom Szwagier,
  * Nina Miolane.

Technical Committee
-------------------
The Technical Committee (TC) members are core developers who have additional
responsibilities to ensure the smooth running of the project. TC members are expected to
participate in strategic planning, and approve changes to the governance model.
The purpose of the TC is to ensure a smooth progress from the big-picture
perspective. Indeed changes that impact the full project require a synthetic
analysis and a consensus that is both explicit and informed. In cases that the
core developer community (which includes the TC members) fails to reach such a
consensus in the required time frame, the TC is the entity to resolve the
issue.
Membership of the TC is by nomination by a core developer. A nomination will
result in discussion which cannot take more than a month and then a vote by
the core developers which will stay open for a week. TC membership votes are
subject to a two-third majority of all cast votes as well as a simple majority
approval of all the current TC members. TC members who do not actively engage
with the TC duties are expected to resign.

The Technical Committee of Geomstats currently consists of:

  * Nina Miolane, 
  * Alice Le Brigant,
  * Xavier Pennec.

Decision Making Process
=======================

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place at the project's monthly meetings and follow-up emails,
and the `issue tracker <https://github.com/geomstats/geomstats/issues>`_.
Occasionally, sensitive discussion occurs on a private list.

Geomstats uses a "consensus seeking" process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
At any point during the discussion, any core-developer can call for a vote, which will
conclude one month from the call for the vote. If no option can gather two thirds of the votes cast, the
decision is escalated to the TC, which in turn will use consensus seeking with
the fallback option of a simple majority vote if no consensus can be found
within a month. This is what we hereafter may refer to as "the decision making
process".

Decisions (in addition to adding core developers and TC membership as above)
are made according to the following rules:

* **Minor Documentation changes**, such as typo fixes, or addition / correction of a
  sentence, but no change of the geomstats.ai landing page or the "about"
  page: Requires +1 by a core developer, no -1 by a core developer (lazy
  consensus), happens on the issue or pull request page. Core developers are
  expected to give "reasonable time" to others to give their opinion on the pull
  request if they are not confident others would agree.

* **Code changes and major documentation changes**
  require +1 by one core developer, no -1 by a core developer (lazy
  consensus), happens on the issue of pull-request page.

* **Changes to the API principles and changes to dependencies or supported
  versions** follow the decision-making process outlined above.

* **Changes to the governance model** use the same decision process outlined above.

If a veto -1 vote is cast on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected using
the decision making procedure outlined above.


Conflict of Interest
====================

It is expected that Geomstats Team Members will be employed at a wide range of companies, 
universities and non-profit organizations. Because of this, it is possible that Members will have 
conflict of interests. Such conflict of interests include, but are not limited to:

  * Financial interests, such as investments, employment or contracting work, outside of Geomstats that may influence their contributions.
  * Access to proprietary information of their employer that could potentially leak into their work with Geomstats.

All members of Geomstats shall disclose to the Technical Committee any conflict of interest they may have. 

Members with a conflict of interest in a particular issue may participate in discussions on that issue, but must recuse themselves from voting on the issue.


Breach
======

Non-compliance with the terms of the governance documents shall be reported to the Technical Committee either through public or private channels as deemed appropriate.

Changing the Governance Documents
=================================

Changes to the governance documents are submitted via a GitHub pull request targeting `Geomstats governance documents <https://github.com/geomstats/geomstats/blob/main/docs/governance.rst>`_. 
The pull request is then refined in response to public comment and review, with the goal being consensus in the community. 
After this open period, a member of the Technical Committee proposes to the core developers that the changes be ratified and the pull request merged (accepting the proposed changes) 
or proposes that the pull request be closed without merging (rejecting the proposed changes). The Technical Committee member should state the final commit hash in the pull request being proposed 
for acceptance or rejection and briefly summarize the pull request. A minimum of 60% of the core developers must vote and at least 2/3 of the votes must be positive to carry out the proposed action 
(fractions of a vote rounded up to the nearest integer).
