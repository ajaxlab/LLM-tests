import { createOpenAI } from '@repo/openai'
import { pdfToText } from '@repo/pdf'

export async function createPdfSummary(path: string): Promise<string> {
  const result = await pdfToText(path)
  const text = result.text.replaceAll('\n', ' ')

  const openai = createOpenAI()

  const response = await openai.responses.create({
    model: 'gpt-5.1',
    temperature: 0.1,
    input: [
      {
        role: 'system',
        content:
          'You are a bot that summarizes the following text in Korean. ' +
          "Read the text below, identify the author's problem awareness " +
          'and argument, and summarize the main points.' +
          'The format should be this.\n\n' +
          '# Title\n\n' +
          "## The author's problem awareness and argument " +
          '(Summarize within 10 sentences)\n\n' +
          '## About the author\n\n' +
          'This is the text.\n\n' +
          text,
      },
    ],
  })

  return response.output_text
}
