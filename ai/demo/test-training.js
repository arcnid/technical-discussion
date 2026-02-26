const puppeteer = require('puppeteer');

async function testTraining() {
    console.log('🚀 Starting Puppeteer test...');

    const browser = await puppeteer.launch({
        headless: false,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();

    // Capture console logs from the page
    page.on('console', msg => {
        const type = msg.type();
        const text = msg.text();
        if (type === 'error') {
            console.error('❌ PAGE ERROR:', text);
        } else if (text.includes('✓') || text.includes('Training') || text.includes('TensorFlow')) {
            console.log('📄 PAGE:', text);
        }
    });

    // Capture page errors
    page.on('pageerror', error => {
        console.error('❌ PAGE EXCEPTION:', error.message);
    });

    try {
        console.log('📂 Navigating to http://localhost:8000...');
        await page.goto('http://localhost:8000', {
            waitUntil: 'networkidle0',
            timeout: 30000
        });

        console.log('✅ Page loaded');

        // Wait for the app to initialize
        await page.waitForSelector('#train-btn', { timeout: 10000 });
        console.log('✅ Train button found');

        // Check if images loaded
        const imageCount = await page.$eval('#image-count', el => el.textContent);
        console.log('📊 Images loaded:', imageCount);

        // Click the train button
        console.log('🔵 Clicking train button...');
        await page.click('#train-btn');

        // Wait a bit for training to start
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Monitor training status
        let lastStatus = '';
        let trainingComplete = false;
        let errorOccurred = false;

        for (let i = 0; i < 60; i++) {
            const status = await page.$eval('#status', el => el.textContent);

            if (status !== lastStatus) {
                console.log('📊 Status:', status);
                lastStatus = status;
            }

            if (status.includes('complete')) {
                trainingComplete = true;
                console.log('✅ TRAINING COMPLETED SUCCESSFULLY!');
                break;
            }

            if (status.includes('failed')) {
                errorOccurred = true;
                console.error('❌ TRAINING FAILED:', status);
                break;
            }

            // Check epoch counter
            try {
                const epoch = await page.$eval('#epoch', el => el.textContent);
                if (epoch !== '0 / 20') {
                    console.log('📈 Epoch:', epoch);
                }
            } catch (e) {
                // Ignore
            }

            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        if (!trainingComplete && !errorOccurred) {
            console.log('⏱️  Training timeout - still running after 60 seconds');
        }

        // Keep browser open for inspection
        console.log('\n🔍 Browser staying open for inspection...');
        console.log('Press Ctrl+C to close');

        // Wait indefinitely
        await new Promise(() => {});

    } catch (error) {
        console.error('❌ Test error:', error.message);
        await browser.close();
        process.exit(1);
    }
}

testTraining();
