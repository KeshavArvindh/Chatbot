import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras
import tensorflow as tf


lemmatizer=WordNetLemmatizer()

json_data = '''
{
    "intents": [
        {
            "tag": "Course Details",
            "patterns": ["How many courses do I have to take per semester?", "What is the typical course load per semester?,"],
            "responses": ["As a fresher, you'll typically enroll in about 4 to 5 courses per semester.", "The standard course load is usually around 4 to 5 courses per semester."]
        },
        {
            "tag": "goodbye",
            "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "bye", "cao", "see ya"],
            "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"]
        },
{
            "tag": "Class Size",
            "patterns": ["What is the average class size for freshmen?", "How many students are usually in a freshman class?"],
            "responses": ["The average class size for freshmen is usually around 30 to 40 students.", "Freshman classes typically consist of around 30 to 40 students on average."]
        },
{
            "tag": "Mandatory Courses",
            "patterns": ["Are there any mandatory courses I have to take during my first year?", "Are certain courses compulsory for first-year students?"],
            "responses": ["Yes, there are usually some core courses that are mandatory for all freshmen to take during their first year.", "Certain core courses are mandatory for all first-year students."]
        },
{
            "tag": "Roommate Allocation",
            "patterns": ["Can I choose my roommate or is it assigned?", "Is roommate selection a choice for students?"],
            "responses": ["In most cases, you'll have the option to choose a roommate or submit preferences for roommates during the hostel allocation process.", "Yes, you can typically choose your roommate or specify preferences during the hostel allocation process."]
        },
{
            "tag": "Hostel Amenities",
            "patterns": ["Are hostels equipped with basic amenities like Wi-Fi and laundry facilities?", "What essential amenities are available in the hostels?"],
            "responses": ["Yes, hostels are equipped with essential amenities including Wi-Fi, laundry facilities, and more.", "Hostels provide essential amenities such as Wi-Fi and laundry facilities for the convenience of students."]
        },
{
            "tag": "Hostel Security",
            "patterns": ["How is the hostel security ensured?", "What measures are taken to ensure hostel security?"],
            "responses": ["Hostel security is maintained through access control systems, CCTV cameras, and security personnel patrolling the premises.", "Access control systems, CCTV surveillance, and regular security patrols ensure the safety and security of the hostels."]
        },
{
            "tag": "Dietary Options",
            "patterns": ["Are there options for vegetarian and non-vegetarian meals in the mess?", "Can I choose between vegetarian and non-vegetarian meals in the mess?"],
            "responses": ["Yes, the mess usually provides both vegetarian and non-vegetarian meal options to cater to different dietary preferences.", "Absolutely, the mess offers both vegetarian and non-vegetarian meal choices to accommodate various dietary preferences."]
        },
{
            "tag": "Mess Timing Flexibility",
            "patterns": ["Is there flexibility in the mess timings to accommodate different schedules?", "Do the mess timings cater to different daily routines?"],
            "responses": ["Yes, mess timings are typically designed to accommodate different schedules and ensure that all students have access to meals.", "Indeed, the mess timings are structured to suit various student schedules and ensure convenient meal access."]
        },
{
            "tag": "Special Dietary Needs",
            "patterns": ["How are dietary restrictions and special meal requirements handled in the mess?", "Can the mess accommodate specific dietary needs, like allergies or preferences?"],
            "responses": ["The mess often caters to various dietary needs, including allergies and preferences, to ensure a diverse range of food options for students.", "Certainly, the mess makes provisions to cater to specific dietary requirements, such as allergies or dietary preferences."]
        },
{
            "tag": "Transportation Accessibility",
            "patterns": ["How is the transportation system around the college campus?", "What transportation options are available around the campus?"],
            "responses": ["The college usually has a transportation system, such as buses, that facilitates commuting to and from the campus and nearby areas.", "Transportation options include college-provided buses and nearby public transportation facilities for convenient travel to and from campus."]
        },
{
            "tag": "Shuttle Service",
            "patterns": ["Is there a shuttle service from the hostel to the academic buildings?", "Do they provide a shuttle service for students from the hostels to the academic area?"],
            "responses": ["Yes, many colleges provide a shuttle service that transports students from the hostels to the academic buildings and vice versa.", "Absolutely, a shuttle service is available to transport students between the hostels and academic buildings for their convenience."]
        },
{
            "tag": "Public Transportation Reliability",
            "patterns": ["How reliable is the public transportation around the college area?", "Can I depend on public transportation for my daily commute to college?"],
            "responses": ["The reliability of public transportation varies, but in general, the college often provides information about nearby public transportation options.", "Public transportation reliability can vary; however, the college typically offers guidance on the nearby public transportation options to assist students."]
        },
{
            "tag": "Professor Availability",
            "patterns": ["How accessible are professors outside of class hours?", "Can I meet with professors outside of their class times?"],
            "responses": ["Professors are usually accessible during their office hours and can also be reached via email for academic inquiries or to schedule appointments.", "Professors make themselves accessible during their designated office hours and can also be contacted via email to address academic concerns."]
        },
{
            "tag": "Academic Guidance",
            "patterns": ["Are there opportunities for academic mentoring or guidance from professors?", "Can professors provide academic advice and mentorship?"],
            "responses": ["Yes, many colleges have mentorship programs where you can seek guidance and advice from experienced professors regarding your academic journey.", "Indeed, there are mentorship programs where professors can provide academic guidance and mentorship to students seeking assistance."]
        },
{
            "tag": "Handling Missed Classes",
            "patterns": ["How do professors typically handle missed classes or assignments due to genuine reasons?", "What is the protocol for handling missed classes or assignments due to unforeseen circumstances?"],
            "responses": ["Professors often have a policy for handling missed classes or assignments due to valid reasons. Communication and discussing the situation with the professor is important.", "Professors usually have established protocols for addressing missed classes or assignments due to legitimate reasons. It's important to communicate with the professor to discuss the situation."]
        },
{
            "tag": "Extracurricular Participation",
            "patterns": ["Can I participate in multiple clubs and organizations simultaneously?", "Is it possible to join more than one club or organization at a time?"],
            "responses": ["Yes, you can join multiple clubs and organizations based on your interests and time management.", "Certainly, you have the opportunity to join multiple clubs and organizations as long as you can manage your time effectively."]
        },
{
            "tag": "International Integration",
            "patterns": ["Are there opportunities for international student integration and support?", "How does the university assist international students in integrating into the community?"],
            "responses": ["Yes, colleges often have programs and support services to help international students integrate into the community and adjust to campus life.", " Absolutely, the university offers programs and support services to aid international students in integrating into the community and acclimating to campus life."]
        },
{
            "tag": " Mental Health Support",
            "patterns": ["How is mental health support provided on campus?", " Are there resources available for mental health assistance?"],
            "responses": ["Colleges usually have counseling services and mental health support available, and they often conduct awareness campaigns to promote mental well-being.", " There are mental health services available on campus, including counseling and support, and the university often raises awareness about mental well-being through campaigns and initiatives."]
        },
        {"tag": "greeting",
   "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day", "whatsup"],
   "responses": ["Hello", "Good to see you", "Hi there, how can I help?"],
   "context": ""
  },

  {"tag": "you",
   "patterns": ["Who are you?", "What are you?", "Who you are?", "what do you do?", "what is your name? "],
   "responses": ["I am a SNUnibot, an AI chatbot of SNUC created by students of SNU Chennai to help you with your queries regarding the college."],
   "context": ""
  },

  {"tag": "thanks",
   "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
   "responses": ["My pleasure", "You're Welcome"],
   "context": ""
  },

  {"tag": "SNUC",
   "patterns": ["snuc", "SNUC", "tell me about the college", "tell me about university", "tell me about snuc","tell me about shiv nadar university", "is college good?", "what do you say about the college", "tell me about shiv nadar university chennai", "give information of shiv nadar university chennai", "whats history of shiv nadar university chennai", "founder of shiv nadar university chennai", "shiv nadar university chennai ranking", "information about shiv nadar university chennai", "when is shiv nadar university chennai founded?"],
   "responses": ["Shiv Nadar University was launched in October 2020, the second university to be set up by the Shiv Nadar Foundation, following Shiv Nadar University, Dadri in Uttar Pradesh. It is one of the best universities in India. For more information, refer this website: https://www.snuchennai.edu.in/"],
   "context": ""
  },

  {"tag": "fees",
   "patterns": ["fees", "What's the fees of AI-DS branch? ", "What's the fees of IOT branch?", "What's the fees of cyber branch?", "What is the fees of first year students?"],
   "responses":["For fee details, click on this website and scroll down: https://www.snuchennaiadmissions.com/"],
   "context": ""    
  },

  {"tag": "placements",
   "patterns": ["placements", "What's the placements details of shiv nadar university chennai? ", "How many pople got placed this year?", "Is placements good?", "what about placements?"],
   "responses": ["Placements at Shiv Nadar University Chennai will be handled by the same team which has churned out excellent placements for SSN over the years. The fact that over 230 organizations visited the SSN campus last year and made over 1200 quality offers is a testimony to the dynamism of the placement team. Google, Microsoft, Atlassian, Adobe, Amazon, Goldman Sachs, Citi, Barclays and PayPal are some of the prominent recruiters that hire top talents from SSN every academic year. For more information, click on this website: https://www.snuchennai.edu.in/placement/"],
   "context": ""
  },
  
  {"tag": "VC",
   "patterns": ["vice chancellor" ,"Who is the Vice Chancellor of Shiv Nadar University Chennai?", "Who is the VC of Shiv Nadar University Chennai?", "Who is the Vice Chancellor of SNU Chennai?", "Who is the VC of SNU Chennai?"],
   "responses": ["Mr. Shri Kumar Battacharya is the Vice- Chancellor of SNU Chennai."],
   "context": ""
  },

  {"tag": "staff",
   "patterns": ["teachers", "faculty", "staff", "What about the staff of Snu chennai? ", "Are teachers good? ", "How many teachers are there in SNU Chennai? ", "How many teachers are there in SNU Chennai? ", "Who are the teachers of snu chennai", "Are teachers good in snu chennai"],
   "responses": ["Teachers are very supportive and will encourage students in all aspects. For staff details, refer this website: https://www.snuchennai.edu.in/faculty/"],
   "context": ""

  }, 
  {"tag": "courses",
   "patterns": ["What are the courses offered? ", "Which courses are there in SNU chennai?", "Which courses are there in snu chennai?", "How many courses are there in snu chennai? ", "How many courses are there in snuc?", "tell me about aids course.", "Tell me about iot course", "tell me about ai-ds course", "tell me about cyber course", "are courses good?"],
   "responses": ["There are totally 3 kinds of courses offered by Shiv Nadar University Chennai. They are 1. B.Tech AI-DS 2. B.Tech IOT 3. B.Tech Cyber Security. For more information, refer this website: https://www.snuchennai.edu.in/academics/"],
   "context": ""
  }, 
  {"tag": "hostels", 
   "patterns": ["Hostels", "How are hostels in snu chennai? ", "are hostels good in college? ", "what kind of rooms will be given for 1st year students?", "what is the fee structure of the hostels?", "how many hostels are there in snu chennai?", "How many members per room? ", "Are hostels safe? ", "Do they charge very more in hostels? ", "How are wardens in hostels?", "Can students choose their rooms? "],
   "responses": ["For 1st year students in SNU Chennai, the hostels are only shared. There are no single rooms for 1st year students. From second year, students can apply for single rooms. The wardens are cool and will be supportive. There will be a daily attendence in the hostel through the biometric scanner machine. For fee details, click on this link and scroll down: https://www.snuchennaiadmissions.com/"]
  },
  {"tag": "facilities",
   "patterns": ["facilities","what are the facilities in the college?", "are facilities good in snu chennai?"],
   "responses": ["The facilities in SNU Chennai are very good. There are many facilities like library, sports, gym, etc. For more information, refer this website: https://www.snuchennai.edu.in/campus-life/"],
   "context": ""
  },

  {"tag": "location",
   "patterns": ["location", "where is the college located?", "where is snu chennai located?", "where is shiv nadar university chennai located?", "where is snu chennai?", "where is shiv nadar university chennai?"],
    "responses": ["Shiv Nadar University Chennai is located on SH-49A in Kalavakkam village near Thiruporur, on the Old Mahabalipuram read, around 20 km south of Chennai in Tamil Nadu, India."],
    "context": ""
  }, 
  {"tag": "admission",
   "patterns": ["admission", "how to get admission in snu chennai?", "how to get admission in shiv nadar university chennai?", "how to get admission in snu chennai?", "how to get admission in shiv nadar university chennai?", "how to get admission in snu chennai?", "how to get admission in shiv nadar university chennai?"],
   "responses": ["For admission details, click on this website: https://www.snuchennaiadmissions.com/"],
   "context": ""
  },
  {"tag": "contact",
   "patterns": ["contact","contacts", "contact number", "what is the contact number of snu chennai?", "what is the contact number of shiv nadar university chennai?", "what is the contact number of snu chennai?", "what is the contact number of shiv nadar university chennai?", "what is the contact number of snu chennai?", "what is the contact number of shiv nadar university chennai?", "what is the email of snu chennai?", "email", "what is the email id of the college"],
   "responses": ["The contact number of Shiv Nadar University Chennai is 044-4743-0000. The email ID of the University is: info@snuchennai.edu.in admissions@snuchennai.edu.in . For more details, refer this website: https://www.snuchennai.edu.in/contact-us/"],
   "context": ""
  },
  {"tag": "school of enginnering",
  "patterns": ["school of engineering", "school of engineering snu chennai", "school of engineering shiv nadar university chennai", "school of engineering snu chennai", "school of engineering shiv nadar university chennai", "school of engineering snu chennai", "school of engineering shiv nadar university chennai"],
  "responses": ["The School of Engineering at Shiv Nadar University Chennai offers B.Tech. programs in three disciplines: Artificial Intelligence and Data Science, Internet of Things and Cyber Security, M.Tech in Artificial Intelligence and Data Science and Full Time PHD program. HOD of School of Enginnering is: Dr. T. Nagarajan. For more information, refer this website: https://www.snuchennai.edu.in/school-of-engineering/"],
  "context": ""
  }, 
  {"tag": "school of humanities and social sciences",
   "patterns": ["school of humanities and social sciences", "school of humanities and social sciences snu chennai", "school of humanities and social sciences shiv nadar university chennai", "school of humanities and social sciences snu chennai", "school of humanities and social sciences shiv nadar university chennai", "school of humanities and social sciences snu chennai", "school of humanities and social sciences shiv nadar university chennai"],
   "responses": ["The School of Sciences and Humanities offers a BSc program in Economics (Data Science) as well as courses that enrich the programs offered by the two other schools. These courses not only provide a scaffolding for the students in the different programs offered at the university, but also help students develop multidisciplinary perspectives. Through these courses, students develop a wider and deeper understanding and also hone their 21st-century skills such as critical thinking, communication, creativity, and collaboration. For more details, refer this website: https://www.snuchennai.edu.in/schools-science-humanities/"],
    "context": ""
  },
  {"tag": "school of commerce and management",
   "patterns": ["school of commerce and management", "school of commerce and management snu chennai", "school of commerce and management shiv nadar university chennai", "school of commerce and management snu chennai", "school of commerce and management shiv nadar university chennai", "school of commerce and management snu chennai", "school of commerce and management shiv nadar university chennai"],
   "responses": ["The School of Commerce and Management offers a BBA program in Business Analytics as well as courses that enrich the programs offered by the two other schools. These courses not only provide a scaffolding for the students in the different programs offered at the university, but also help students develop multidisciplinary perspectives. Through these courses, students develop a wider and deeper understanding and also hone their 21st-century skills such as critical thinking, communication, creativity, and collaboration. For more details, refer this website: https://www.snuchennai.edu.in/school-of-commerce-management/"],
    "context": ""
  },
  {"tag": "clubs",
   "patterns": ["clubs", "clubs in snu chennai", "clubs in shiv nadar university chennai", "clubs in snu chennai", "clubs in shiv nadar university chennai", "clubs in snu chennai", "clubs in shiv nadar university chennai"],
   "responses": ["There are these many clubs"],
    "context": ""
  },
  {"tag": "curriculum",
   "patterns": ["syllabus", "curriculum of ai-ds", "curriculum", "curriculum in snu chennai", "curriculum in shiv nadar university chennai","curriculum in college", "curriculum in shiv nadar university chennai","what is the curriculum of snu chennai", "what is the syllabus of snu chennai","is syllabus good in snu chennai", "is curriculum good in snu chennai"],
   "responses":["The curriculum of Shiv Nadar University Chennai is very good. It is designed in such a way that it helps students to learn and understand the concepts easily. For more details, click on this website and go to the respective branch's curriculum: https://www.snuchennai.edu.in/schools-engineering/"],
    "context": ""

  }
    ]
}

'''

intents = json.loads(json_data)

words=[]
documents=[]
classes=[]
ignore_letters=['?',',','.','!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row]) 


random.shuffle(training)
training = np.array(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

model= tf.keras.Sequential()
model.add(tf.keras.layers.Dense(152,input_shape=(len(train_x[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(76,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd=tf.keras.optimizers.legacy.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist=model.fit(np.array(train_x),np.array(train_y),epochs=150,batch_size=5,verbose=1)
model.save('chatbotmodel.h5',hist)
print('Done')