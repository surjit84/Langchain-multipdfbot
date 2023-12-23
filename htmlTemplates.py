css='''
<style>
.chat-meassge{
  padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-meassge.user{
  background-color:#2b313e
}
.chat-meassge.bot {
  background-color:#475063
}
.chat-meassge .avatar {
  width: 15%
}
.chat-meassge .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-meassge .message {
  width: 85%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="https://www.kasandbox.org/programming-images/avatars/marcimus.png">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
  <div class="avatar">
    <img src="https://www.kasandbox.org/programming-images/avatars/spunky-sam.png">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''