import { pdfToText } from '@repo/pdf';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
  const pdfPath = join(__dirname, 'sample.pdf');

  console.log('PDF 파일 읽는 중:', pdfPath);

  const result = await pdfToText(pdfPath);

  console.log('=== PDF 정보 ===');
  console.log('페이지 수:', result.numPages);
  console.log('\n=== 추출된 텍스트 ===');
  console.log(result.text);
}

main().catch(console.error);
