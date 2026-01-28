import { speechToTranslation } from '@repo/ai-util'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

async function main() {
  const text = await speechToTranslation(join(__dirname, 'sample.mp3'))
  console.log(text)
}

main().catch(console.error)
