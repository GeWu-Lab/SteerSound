import unittest
from steersound.instructions import EXAMPLES,MODEL_PREFIX,parse_instruction,model_prompt,strip_model_prefix

class InstructionsTest(unittest.TestCase):
 def test_all_six_tasks(self):
  for task,text in EXAMPLES.items():self.assertEqual(parse_instruction(text).task,task)
 def test_amounts(self):
  self.assertEqual(parse_instruction(EXAMPLES['volume']).change_vol,3)
  self.assertEqual(parse_instruction(EXAMPLES['pitch']).change_pitch,-2)
  self.assertEqual(parse_instruction('Reduce the volume of the sound of rain by 2.5.').change_vol,-2.5)
 def test_prefix_is_internal_and_idempotent(self):
  plain=EXAMPLES['remove']
  self.assertEqual(model_prompt(plain),model_prompt(MODEL_PREFIX+'\n'+plain))
  self.assertEqual(model_prompt(model_prompt(plain)),model_prompt(plain))
  self.assertEqual(model_prompt(''),'')
  self.assertEqual(strip_model_prefix(MODEL_PREFIX+'\n'+plain),plain)
 def test_no_guessed_amounts_or_silent_clipping(self):
  for text in ('Make it better.','Increase the volume of rain.','Reduce the pitch of rain by 9 semitones.',
               'Increase the volume of rain by 6 dB.','Increase the pitch of rain by 1.2 semitones.',
               'Increase the pitch and volume of rain by 2.0.','Reduce the pitch of rain by -2 semitones.','Add the sound of rain and remove the sound of a dog.',
               'Increase the volume of rain by 3 percent.'):
   with self.subTest(text=text),self.assertRaises(ValueError):parse_instruction(text)
 def test_sound_caption_is_not_a_second_operation(self):
  self.assertEqual(parse_instruction('Generate the sound of whistling sounds start low and increase in pitch.').task,'generate')
 def test_native_conditioning_text(self):
  self.assertEqual(parse_instruction(EXAMPLES['pitch']).conditioning_text,'Reduce the pitch of the sound of a dog barking by 2.0.')
 def test_time_range(self):
  result=parse_instruction('Remove the sound of rain from 1.0 to 3.0 seconds.')
  self.assertEqual((result.start_time,result.end_time),(1,3))
if __name__=='__main__':unittest.main()
