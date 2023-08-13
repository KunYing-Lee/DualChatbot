import streamlit as st
import time
from streamlit_chat import message
from gtts import gTTS
from io import BytesIO
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Add a sidebar input for API keys
api_key = st.sidebar.text_input("Enter your OpenAI API key")

class Chatbot:
    
    def __init__(self, enginecd):
        os.environ["OPENAI_API_KEY"] = api_key
        
        if engine == 'OpenAI':
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )


        else:
            raise KeyError("Currently unsupported chat model type!")
        
        # Instantiate memory
        self.memory = ConversationBufferMemory(return_messages=True)



    def instruct(self, role, oppo_role, language, scenario, 
                 session_length, proficiency_level, 
                 learning_mode, starter=False):
        
        # Define language settings
        self.role = role
        self.oppo_role = oppo_role
        self.language = language
        self.scenario = scenario
        self.session_length = session_length
        self.proficiency_level = proficiency_level
        self.learning_mode = learning_mode
        self.starter = starter
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])
        
        # Create conversation chain
        self.conversation = ConversationChain(memory=self.memory, prompt=prompt, 
                                              llm=self.llm, verbose=False)
        


    def _specify_system_message(self):
        
        # Determine the number of exchanges between two bots
        exchange_counts_dict = {
            'Short': {'Conversation': 8, 'Debate': 4},
            'Long': {'Conversation': 16, 'Debate': 8}
        }
        exchange_counts = exchange_counts_dict[self.session_length][self.learning_mode]
        
        # Determine number of arguments in one debate round
        argument_num_dict = {
            'Beginner': 4,
            'Intermediate': 6,
            'Advanced': 8
        }        
        
        # Determine language complexity
        if self.proficiency_level == 'Beginner':
            lang_requirement = """use as basic and simple vocabulary and
            sentence structures as possible. Must avoid idioms, slang, 
            and complex grammatical constructs."""
        
        elif self.proficiency_level == 'Intermediate':
            lang_requirement = """use a wider range of vocabulary and a variety of sentence structures. 
            You can include some idioms and colloquial expressions, 
            but avoid highly technical language or complex literary expressions."""
        
        elif self.proficiency_level == 'Advanced':
            lang_requirement = """use sophisticated vocabulary, complex sentence structures, idioms, 
            colloquial expressions, and technical language where appropriate."""

        else:
            raise KeyError('Currently unsupported proficiency level!')
    
        
        # Compile bot instructions 
        if self.learning_mode == 'Conversation':
            prompt = f"""You are an AI that is good at role-playing. 
            You are simulating a typical conversation happened {self.scenario}. 
            In this scenario, you are playing as a {self.role['name']} {self.role['action']}, speaking to a 
            {self.oppo_role['name']} {self.oppo_role['action']}.
            Your conversation should only be conducted in {self.language}. Do not translate.
            This simulated {self.learning_mode} is designed for {self.language} language learners to learn real-life 
            conversations in {self.language}. You should assume the learners' proficiency level in 
            {self.language} is {self.proficiency_level}. Therefore, you should {lang_requirement}.
            You should finish the conversation within {exchange_counts} exchanges with the {self.oppo_role['name']}. 
            Make your conversation with {self.oppo_role['name']} natural and typical in the considered scenario in 
            {self.language} cultural."""
        
        elif self.learning_mode == 'Debate':
            prompt = f"""You are an AI that is good at debating. 
            You are now engaged in a debate with the following topic: {self.scenario}. 
            In this debate, you are taking on the role of a {self.role['name']}. 
            Always remember your stances in the debate.
            Your debate should only be conducted in {self.language}. Do not translate.
            This simulated debate is designed for {self.language} language learners to 
            learn {self.language}. You should assume the learners' proficiency level in {self.language} 
            is {self.proficiency_level}. Therefore, you should {lang_requirement}.
            You will exchange opinions with another AI (who plays the {self.oppo_role['name']} role) 
            {exchange_counts} times. 
            Everytime you speak, you can only speak no more than 
            {argument_num_dict[self.proficiency_level]} sentences."""
        
        else:
            raise KeyError('Currently unsupported learning mode!')
        
        # Give bot instructions
        if self.starter:
            # In case the current bot is the first one to speak
            prompt += f"You are leading the {self.learning_mode}. \n"
        
        else:
            # In case the current bot is the second one to speak
            prompt += f"Wait for the {self.oppo_role['name']}'s statement."
        
        return prompt
    



class DualChatbot:
    
    def __init__(self, engine, role_dict, language, scenario, proficiency_level, 
                 learning_mode, session_length):
    
        # Instantiate two chatbots
        self.engine = engine
        self.proficiency_level = proficiency_level
        self.language = language
        self.chatbots = role_dict
        for k in role_dict.keys():
            self.chatbots[k].update({'chatbot': Chatbot(engine)})
            
        # Assigning roles for two chatbots
        self.chatbots['role1']['chatbot'].instruct(role=self.chatbots['role1'], 
                                                   oppo_role=self.chatbots['role2'], 
                                                   language=language, scenario=scenario, 
                                                   session_length=session_length, 
                                                   proficiency_level=proficiency_level, 
                                                   learning_mode=learning_mode, starter=True)
        
        self.chatbots['role2']['chatbot'].instruct(role=self.chatbots['role2'], 
                                                   oppo_role=self.chatbots['role1'], 
                                                   language=language, scenario=scenario, 
                                                   session_length=session_length, 
                                                   proficiency_level=proficiency_level, 
                                                   learning_mode=learning_mode, starter=False) 

        
        # Add session length
        self.session_length = session_length

        # Prepare conversation
        self._reset_conversation_history()



    def step(self):
       
        # Chatbot1 speaks
        output1 = self.chatbots['role1']['chatbot'].conversation.predict(input=self.input1)
        self.conversation_history.append({"bot": self.chatbots['role1']['name'], "text": output1})
            
        # Pass output of chatbot1 as input to chatbot2
        self.input2 = output1
        
        # Chatbot2 speaks
        output2 = self.chatbots['role2']['chatbot'].conversation.predict(input=self.input2)
        self.conversation_history.append({"bot": self.chatbots['role2']['name'], "text": output2})
        
        # Pass output of chatbot2 as input to chatbot1
        self.input1 = output2

        # Translate responses
        translate1 = self.translate(output1)
        translate2 = self.translate(output2)

        return output1, output2, translate1, translate2
    


    def translate(self, message):
        
        if self.language == 'English':
            # No translation performed
            translation = 'Translation: ' + message

        else:
            # Instantiate translator
            if self.engine == 'OpenAI':
                
                self.translator = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.7
                )

            else:
                raise KeyError("Currently unsupported translation model type!")
            
            # Specify instruction
            instruction = """Translate the following sentence from {src_lang} 
            (source language) to {trg_lang} (target language).
            Here is the sentence in source language: \n
            {src_input}."""

            prompt = PromptTemplate(
                input_variables=["src_lang", "trg_lang", "src_input"],
                template=instruction,
            )

            # Create a language chain
            translator_chain = LLMChain(llm=self.translator, prompt=prompt)
            translation = translator_chain.predict(src_lang=self.language,
                                                trg_lang="English",
                                                src_input=message)

        return translation
    


    def summary(self, script):
        
        # Instantiate summary bot
        if self.engine == 'OpenAI':
            
            self.summary_bot = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )

        else:
            raise KeyError("Currently unsupported summary model type!")
        

        # Specify instruction
        instruction = """The following text is a simulated conversation in 
        {src_lang}. The goal of this text is to aid {src_lang} learners to learn
        real-life usage of {src_lang}. Therefore, your task is to summarize the key 
        learning points based on the given text. Specifically, you should summarize 
        the key vocabulary, grammar points, and function phrases that could be important 
        for students learning {src_lang}. Your summary should be conducted in English, but
        use examples from the text in the original language where appropriate.
        Remember your target students have a proficiency level of 
        {proficiency} in {src_lang}. You summarization must match with their 
        proficiency level. 

        The conversation is: \n
        {script}."""

        prompt = PromptTemplate(
            input_variables=["src_lang", "proficiency", "script"],
            template=instruction,
        )

        # Create a language chain
        summary_chain = LLMChain(llm=self.summary_bot, prompt=prompt)
        summary = summary_chain.predict(src_lang=self.language,
                                        proficiency=self.proficiency_level,
                                        script=script)
        
        return summary
    


    def _reset_conversation_history(self):
        """Reset the conversation history.
        """    
        # Placeholder for conversation history
        self.conversation_history = []

        # Inputs for two chatbots
        self.input1 = "Start the conversation."
        self.input2 = "" 

# Define the language learning settings
LANGUAGES = ['English', 'German', 'Spanish', 'French', 'Chinese']
SESSION_LENGTHS = ['Short', 'Long']
PROFICIENCY_LEVELS = ['Beginner', 'Intermediate', 'Advanced']
MAX_EXCHANGE_COUNTS = {
    'Short': {'Conversation': 8, 'Debate': 4},
    'Long': {'Conversation': 16, 'Debate': 8}
}
AUDIO_SPEECH = {
    'English': 'en',
    'German': 'de',
    'Spanish': 'es',
    'French': 'fr',
    'Chinese': 'zn'
}
AVATAR_SEED = [123, 42]

# Define backbone llm
engine = 'OpenAI'

# Set the title of the app
st.title('Dual Chatbot Language Learning')

# Set the description of the app
st.markdown("""
This app generates conversation or debate scripts to aid in language learning 🎯 

Choose your desired settings and press 'Generate' to start 🚀
""")

# Add a selectbox for learning mode
learning_mode = st.sidebar.selectbox('Learning Mode 📖', ('Conversation', 'Debate'))

if learning_mode == 'Conversation':
    role1 = st.sidebar.text_input('Role 1 🎭')
    action1 = st.sidebar.text_input('Action 1 🗣️')
    role2 = st.sidebar.text_input('Role 2 🎭')
    action2 = st.sidebar.text_input('Action 2 🗣️')
    scenario = st.sidebar.text_input('Scenario 🎥')
    time_delay = 2

    # Configure role dictionary
    role_dict = {
        'role1': {'name': role1, 'action': action1},
        'role2': {'name': role2, 'action': action2}
    }

else:
    scenario = st.sidebar.text_input('Debate Topic 💬')

    # Configure role dictionary
    role_dict = {
        'role1': {'name': 'Proponent'},
        'role2': {'name': 'Opponent'}
    }
    time_delay = 5

language = st.sidebar.selectbox('Target Language 🔤', LANGUAGES)
session_length = st.sidebar.selectbox('Session Length ⏰', SESSION_LENGTHS)
proficiency_level = st.sidebar.selectbox('Proficiency Level 🏆', PROFICIENCY_LEVELS)


if "bot1_mesg" not in st.session_state:
    st.session_state["bot1_mesg"] = []

if "bot2_mesg" not in st.session_state:
    st.session_state["bot2_mesg"] = []

if 'batch_flag' not in st.session_state:
    st.session_state["batch_flag"] = False

if 'translate_flag' not in st.session_state:
    st.session_state["translate_flag"] = False

if 'audio_flag' not in st.session_state:
    st.session_state["audio_flag"] = False

if 'message_counter' not in st.session_state:
    st.session_state["message_counter"] = 0


def show_messages(mesg_1, mesg_2, message_counter,
                  time_delay, batch=False, audio=False,
                  translation=False):
    
    for i, mesg in enumerate([mesg_1, mesg_2]):
        # Show original exchange ()
        message(f"{mesg['content']}", is_user=i==1, avatar_style="bottts", 
                seed=AVATAR_SEED[i],
                key=message_counter)
        message_counter += 1
        
        # Mimic time interval between conversations
        # (this time delay only appears when generating 
        # the conversation script for the first time)
        if not batch:
            time.sleep(time_delay)

        # Show translated exchange
        if translation:
            message(f"{mesg['translation']}", is_user=i==1, avatar_style="bottts", 
                    seed=AVATAR_SEED[i], 
                    key=message_counter)
            message_counter += 1

        # Append audio to the exchange
        if audio:
            tts = gTTS(text=mesg['content'], lang=AUDIO_SPEECH[language])  
            sound_file = BytesIO()
            tts.write_to_fp(sound_file)
            st.audio(sound_file)

    return message_counter


# Define the button layout at the beginning
translate_col, original_col, audio_col = st.columns(3)

# Create the conversation container
conversation_container = st.container()

if 'dual_chatbots' not in st.session_state:

    if st.sidebar.button('Generate'):
        
        # Add flag to indicate if this is the first time running the script
        st.session_state["first_time_exec"] = True 

        with conversation_container:
            if learning_mode == 'Conversation':
                st.write(f"""#### The following conversation happens between 
                                {role1} and {role2} {scenario} 🎭""")

            else:
                st.write(f"""#### Debate 💬: {scenario}""")

            # Instantiate dual-chatbot system
            dual_chatbots = DualChatbot(engine, role_dict, language, scenario,
                                        proficiency_level, learning_mode, session_length)
            st.session_state['dual_chatbots'] = dual_chatbots
            
            # Start exchanges
            for _ in range(MAX_EXCHANGE_COUNTS[session_length][learning_mode]):
                output1, output2, translate1, translate2 = dual_chatbots.step()

                mesg_1 = {"role": dual_chatbots.chatbots['role1']['name'], 
                        "content": output1, "translation": translate1}
                mesg_2 = {"role": dual_chatbots.chatbots['role2']['name'], 
                        "content": output2, "translation": translate2}
                
                new_count = show_messages(mesg_1, mesg_2, 
                                          st.session_state["message_counter"],
                                          time_delay=time_delay, batch=False,
                                          audio=False, translation=False)
                st.session_state["message_counter"] = new_count

                # Update session state
                st.session_state.bot1_mesg.append(mesg_1)
                st.session_state.bot2_mesg.append(mesg_2)
                


if 'dual_chatbots' in st.session_state:  

    # Show translation 
    if translate_col.button('Translate to English'):
        st.session_state['translate_flag'] = True
        st.session_state['batch_flag'] = True

    # Show original text
    if original_col.button('Show original'):
        st.session_state['translate_flag'] = False
        st.session_state['batch_flag'] = True

    # Append audio
    if audio_col.button('Play audio'):
        st.session_state['audio_flag'] = True
        st.session_state['batch_flag'] = True

    # Retrieve generated conversation & chatbots
    mesg1_list = st.session_state.bot1_mesg
    mesg2_list = st.session_state.bot2_mesg
    dual_chatbots = st.session_state['dual_chatbots']
    
    # Control message appear
    if st.session_state["first_time_exec"]:
        st.session_state['first_time_exec'] = False
    
    else:
        # Show complete message
        with conversation_container:
            
            if learning_mode == 'Conversation':
                st.write(f"""#### {role1} and {role2} {scenario} 🎭""")

            else:
                st.write(f"""#### Debate 💬: {scenario}""")
        
            for mesg_1, mesg_2 in zip(mesg1_list, mesg2_list):
                new_count = show_messages(mesg_1, mesg_2, 
                                        st.session_state["message_counter"],
                                        time_delay=time_delay,
                                        batch=st.session_state['batch_flag'],
                                        audio=st.session_state['audio_flag'],
                                        translation=st.session_state['translate_flag'])
                st.session_state["message_counter"] = new_count
    

    # Create summary for key learning points
    summary_expander = st.expander('Key Learning Points')
    scripts = []
    for mesg_1, mesg_2 in zip(mesg1_list, mesg2_list):
        for i, mesg in enumerate([mesg_1, mesg_2]):
            scripts.append(mesg['role'] + ': ' + mesg['content'])
    
    # Compile summary
    if "summary" not in st.session_state:
        summary = dual_chatbots.summary(scripts)
        st.session_state["summary"] = summary
    else:
        summary = st.session_state["summary"]
    
    with summary_expander:
        st.markdown(f"**Here is the learning summary:**")
        st.write(summary)
