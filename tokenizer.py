import nltk
import hazm as hz

#At first step after installing the "nltk" package you should download the nltk files using this code
#nltk.download()

sample_text_en = '''The narrator, an airplane pilot, crashes in the Sahara desert." \
              " The crash badly damages his airplane and leaves the narrator with very little food or water." \
              " As he is worrying over his predicament, he is approached by the little prince," \
              " a very serious little blond boy who asks the narrator to draw him a sheep. " \
              "The narrator obliges, and the two become friends. " \
              "The pilot learns that the little prince comes from a small planet that the little prince calls Asteroid 325 " \
              "but that people on Earth call Asteroid B-612. The little prince took great care of this planet, " \
              "preventing any bad seeds from growing and making sure it was never overrun by baobab trees. " \
              "One day, a mysterious rose sprouted on the planet and the little prince fell in love with it. " \
              "But when he caught the rose in a lie one day, he decided that he could not trust her anymore. " \
              "He grew lonely and decided to leave. Despite a last-minute reconciliation with the rose, " \
              "the prince set out to explore other planets and cure his loneliness.'''

sample_text_fa = '''داستان کتاب شازده کوچولو درباره‌ی ملاقات خلبانی با یک موجود کوچولوی دوست‌داشتنی است. این خلبان همان نویسنده‌ی کتاب آنتوان سنت اگزوپری است، او در یکی از سفرهایش به آفریقا، در صحرای آفریقا سقوط می‌کند و در آن‌جا شازده کوچولو را ملاقات می‌کند. داستان شاعرانه و عاشقانه شازده کوچولو از جایی شروع می‌شود که در سیاره‌ ب 612 که سیاره‌ی شازده کوچولو است گیاهی متفاوت در بین علف‌ها رشد می‌کند. گلی که در این داستان نقش معشوق را بازی می‌کند. شازده کوچولو که عاشق گل سرخ می‌شود در مسیر سفر قرار می‌گیرد، سفری که در جست‌وجوی دوست به زمین می‌رسد.  شازده کوچولو در سفرش با ساکنان هفت سیارک همنشین می‌شود و از آدم‌های هر سیاره حقیقت‌هایی درباره‌ی زندگی می‌آموزد.

اگرچه عمر اگزوپری کم بود و از چهل و چهار تجاوز نکرد؛ اما همین فرصت کوتاه باعث شد تا او آثار ارزشمندی مثل: «هوانورد»، «زمین انسان‌ها»، «نامه یک گروگان» و «پرواز شبانه» را از خود به جا بگذارد. اگزوپری، نه‌تنها خلبانی شجاع، وطن‌پرست و مبارزی ضد فاشیسم بود؛ بلکه تجربه‌های زندگی‌اش به‌عنوان خلبان در رمان‌ها و داستان‌های لطیف و خیال‌انگیز  او منعکس شده است.

 
طرح اصلی داستان شازده کوچولو

نویسنده برای طراحی این داستان سراغ تضاد بین بزرگسالی و کودکی رفته است. تقاوتی که بین درک جهان در دنیای بزرگسالی و کودکی است به عنوان عنصر اصلی در این داستان بارها مورد اشاره قرار گرفته است. نویسنده بارها و بارها راوی - خلبان،‌آدم بزرگسال- را در تقابل با شازده کوچولو - کودک - قرار می‌دهد و مدام حماقت‌ها، تعصب‌ها و نادیده‌گیری‌های دنیای بزرگسالی را مورد انتقاد قرار می‌دهد. 

راوی با بحث در مورد ماهیت بزرگسالان و ناتوانی آنها در درک «چیزهای مهم» شروع می کند. به عنوان آزمایشی برای تعیین اینکه آیا یک بزرگ‌تر به اندازه یک کودک باهوش است یا خیر، او تصویر یک مار بوآ را نشان می‌دهد که یک فیل را خورده است. در پاسخ به این پرسش که: این تصویر چیست؟ اغلب بزرگسالان به سادگی پاسخ می‌دهند؛ کلاه! در واقع اولین‌ و مهمترین چیزی که به نظرشان می‌آید. اما جریان به کل چیز دیگری است. 

در ادامه داستان نیز راوی که خلبان یک هواپیما است و هواپیمایش در صحرایی دور دچار سانحه شده بارها و بارها در تقابل با شازده کوچولو قرار می‌گیرد. شازده کوچولو داستان سفرش به سیاره‌های دیگر را تعریف می‌‌کند و از اینکه چقدر روابط دنیای آدم بزرگها برایش عجیب است می‌گوید. او در سفرش با این شخصیت‌ها ملاقات کرده است:
پادشاهی بدون رعیت، که فقط دستوراتی صادر می کند که مطمئن به آنها عمل می‌شود، مانند فرمان غروب خورشید در غروب خورشید!
مردی مغرور که فقط ستایشی را می‌خواهد که ناشی از تحسین باشد و تحسین برانگیزترین فرد در سیاره غیر مسکونی خود باشد.
مستی که شراب می‌نوشد تا  تا شرم نوشیدنش را فراموش کند.
تاجری که از زیبایی ستاره‌ها چشم‌پوشی می‌کند و در عوض بی‌وقفه آنها را می‌شمارد و فهرست می‌کند تا همه آنها را تبدیل به عدد کرده و به مالکیت خود درآورد. 

و فانوس‌بانی که با هر بار شب و روز شدن در سیاره‌اش فانوسی را روشن و خاموش می‌کند. 

در تمام این شخصیت‌ها آنچه دستمایه نویسنده قرار گرفته نمایش پوچی، تکراری و بی‌ذوق بودن دنیای بزرگسالی در تقابل با دنیای کودکی است. اگزوپری خواننده را به برگشت به همان دنیای خالص و ساده کودکی دعوت می‌کند که برای گل‌ها و طبعیت و زیبایی‌های کوچک دنیا اهمیت قائل است.'''

#tokenize as sentences in english
sentences_en = nltk.sent_tokenize(sample_text_en)
print(sentences_en)

#tokenize az words in english
words_en = nltk.word_tokenize(sample_text_en)
print(words_en)


#same tokenize in persian using hazm library
sentences_fa = hz.sent_tokenize(sample_text_fa)
print(sentences_fa)

words_fa = hz.word_tokenize(sample_text_fa)
print(words_fa)


#normalize the text
normalize = hz.Normilizer()
normal_text_fa  = normalize.normalizer(sample_text_fa)