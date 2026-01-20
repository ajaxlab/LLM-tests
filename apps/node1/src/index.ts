import '@repo/env'

import { OpenAI } from 'openai'

const openai = new OpenAI({
  apiKey: process.env.API_KEY,
})

const Q = 'Where was the most recent Olympic Games held?'

const response = openai.responses.create({
  model: 'gpt-5.1',
  temperature: 0.1,
  input: [
    {
      role: 'system',
      content: 'You are a helpful assistant.',
    },
    { role: 'user', content: Q },
  ],
})

console.log('Q:', Q)

response.then((result) => console.log(result))
