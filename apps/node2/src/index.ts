import { createPdfSummary } from '@repo/ai-util'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

async function main() {
  const pdfPath = join(__dirname, 'sample.pdf')

  console.log('Summarizing...')

  const summary = await createPdfSummary(pdfPath)
  console.log(summary)
}

main().catch(console.error)
