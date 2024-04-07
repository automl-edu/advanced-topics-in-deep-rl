---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

# For courses with a single offering in the _config.yml,
# uncomment the following line and comment out the course-multi line

#layout: course-single
layout: course-multi
---

# <a name="description">Overview</a>

{{ site.description }}

## <a name="goals">Learning Goals</a>

Upon completing this course, our goal is for you to be able to:

* Have an overview of different layers of RL approaches:
  * State-of-the-art RL algorithms & exploration strategies
  * AutoRL methods for configuring RL agents reliably
  * Meta-RL approaches 
  * Algorithm discovery ideas
* Read and understand current research in the field of RL
* Develop your own RL research projects and ideas
* Present and discuss your ideas with others
* Refine your implementation & experimentation capabilities

## <a name="resources">Resources</a>

{% include resources.html content=site.resources %}

## <a name="additional-resources">Optional Resources</a>

{% include resources.html content=site.extra-resources %}

# Coursework

Each student has **two abscence days** to spend throughout the semester as they wish.
Simply inform the instructor any time *prior* to the lecture; if you miss more than two days, 
you will have to provide documentation for your abscences (e.g. a sick note) and/or we'll discuss how you can make up the workload.
If you're absent for a longer amount of time and aren't sure if that's okay and what kind of documentation you can provide, 
please come talk to us! Often we can figure out the best way of handling longer abscences together in advance. 

## <a name="labs">The Lecture</a>

The first component of this course is a weekly lecture. 
This is quite similar to a standard lecture our agenda will look like this:

1. Standup round: are there any questions from last session or the seminar? (max. 10 minutes)
2. Lecturer Input 1 (35 minutes)
3. 5 minute break 
4. Lecturer Input 2 (30 minutes)
5. Finishing round: is everything clear? Was anything missing for you? Do you have questions about potential applications, related work, future projects, etc.? (max. 10 minutes)

That means you'll have just over an hour where the week's lecturer will tell you about a specific topic in a traditional lecture style.
This is the course knowledge we'll expect you to know come end of term and that you'll likely need for your project and seminar ideas. 
The lecture is also a great place to ask high-level questions, e.g. about papers you've come across, issues you have with the lecture's topic or about anything else you'd like to know.
More specific questions might be better answered in the *seminar*.

## <a name="projects">The Seminar</a>

Our second weekly session will be a seminar.
Each of you will give a brief presentation of a maxiumum of 3 minutes and 2 slides (we will stop you after 2 slides)! 
After that, there'll be two minutes for questions and remarks.
The presentations will always refer to one of three target settings we'll define in week one: continual learning, offline RL and multi-agent learning.
This part of the seminar should help you practice brainstorming how to improve these settings with approaches and methods we've heard about even if they're not built for our domains orginally.
The content of this presentation should refer to our current topic and provide some insight into what you're curious about, e.g.:

* an experiment you tried to reproduce from a paper on our three target settings
* an idea of how you could apply an existing method to our settings (and why it would work)
* a completely new idea that came to you about how we can improve on our environments

These aren't graded, should be short and don't need to work perfectly. 
It's about brainstorming together, discussing new ideas and giving feedback to each other.
We expect your ideas to be incomplete and your experiments to have flaws at times, our expectation is that you're able to tell us what went wrong and how you'd move forward.
Plus: you can use all the material you accumulate in your exam!

## <a name="exams">The Exam</a>

We'll have an exam at the end of the semester when the lectures have finished. 
You'll be tested on two learning goals:

1. How well can you develop and present a RL rearch project?
2. How well do you know the content of the lecture?

For the first part, you'll present us a research proposal of your choice for 10 minutes. 
You can re-use all the material and experiments from the lecture you like for this.
What we will be looking for:

- Is it an interesting and novel idea?
- Would it be realistic to tackle in the timeframe of roughly 6-8 months for one person on university compute budgets?
- Do you reference related approaches and how they compare?
- Can you provide indicators of how well your idea will work (e.g. some toy experiments)?
- What would be a realistic timeplan to realize your idea?

This means ideally you'll have read related papers, developed an idea, implemented a small version of it and get us some results on what works and what doesn't
Then you'll tell us how you could expand upon it within a few months to make it a conference paper.
**Important: don't overdo it! Take a small idea you can actually work with and don't try to solve RL**

In the second part, we'll ask you questions about topics from the lecture. 
We expect you'll be able to tell us about content from the slides and our lecture discussions here, it's not necessary to know details from the seminar sessions.

## <a name="scale">Grading</a>

You'll receive a grade based on your exam. 
For that, we'll ask you for a short statement of about 2 minutes how you would grade your project (you can also add a sentence about how you think you did on the questions).
Obviously we won't accept proposed grades that are way off from our perception, but if you make a convincing case, we'll try to match you proposal.
The purpose of this is for you to reflect on your work, how much time and thought went into it, how much you learned and how satisfied you are with the project.
Since this lecture's goal is to prepare you for projects in RL research, evaluating your work is a valuable skill as well.
Needless to say, when it comes to the questions, your input will likely factor much less since you're not an expert in the topic quite yet, at least not in the way you'll be an expert on your project.
Trying to cheat this process by providing a very high estimate of your work might backfire, so we suggest you're honest with us and yourself.

