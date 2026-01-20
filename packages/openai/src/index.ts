import { OpenAI } from 'openai';

export function createOpenAI(): OpenAI {
  return new OpenAI({
    apiKey: process.env.API_KEY,
  });
}
