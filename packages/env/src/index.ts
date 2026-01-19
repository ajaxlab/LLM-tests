import { config } from 'dotenv';

// 루트 .env 파일 경로 (process.cwd()는 항상 모노레포 루트)
config({ path: '../../.env' });
