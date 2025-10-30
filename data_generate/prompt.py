DAILY_PROMPT = """
Now I want you to generate your own schedule for today.(today is {day}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes, routines or habits relate to your behavior decisions.
2. I want to limit your total number of events in a day to {day}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

"""