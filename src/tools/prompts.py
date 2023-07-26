PROMPTS = {
    'recaption': 
        [
        {"role": "system", "content": "You are a disciplined automatic content annotator. Instructions will be provided to you so you can perform your task."},
        {"role": "user", "content": """
        Hi! I am trying to predict the memorability score of a video using its text description, or caption. Memorability is defined as the ability of a video to be remembered over time (a memorability score close to 1 will represent a video that is likely to be remember whilst lower scores would mean an easily forgettable video). In particular, it is related with the portion of annotators that are able to correctly remember watching the video after some lag in a memory game. All videos are 3 seconds long and are "home videos", meaning that their quality is not very high. This are some examples:

        Example 1:

        - Memorability score: 0.99
        - Caption: A person demonstrates how to fold a dollar to look like a chicken laying an egg.
        - Actions: dropping, folding, falling, crafting
        Example 2:
        - Memorability score: 0.47
        - Caption: An obscure video of clouds creeping in into the city while cars drive by.
        - Actions: heading, driving

        I would like you to paraphrase the following video caption in a way that is more aligned with the memorability task. You must describe the video in a concise and accurate way but using a vocabulary that is aligned with how memorable you think the video might be, without explicitly saying it. Use a single sentence. This is your first example:

        - Caption: {caption}
        - Actions: {actions}
        - Alternate caption:
"""}
    ],
    'mem_pred': [
    {"role": "system", "content": "You are an expert in memorability and your goal is to predict a memorability score from text descriptions. Instructions will be provided to you so you can perform your task. You should output just numbers, no text."},
    {"role": "user", "content": """
    Hi! I am trying to predict the memorability score of a video using its text description, or caption. Memorability is defined as the ability of a video to be remembered over time (a memorability score close to 1 will represent a video that is likely to be remember whilst lower scores would mean an easily forgettable video). In particular, it is related with the portion of annotators that are able to correctly remember watching the video after some lag in a memory game. All videos are 3 seconds long and are "home videos", meaning that their quality is not very high. This are some examples:

    Example 1:

    - Memorability score: 0.99
    - Caption: A person demonstrates how to fold a dollar to look like a chicken laying an egg.
    - Actions: dropping, folding, falling, crafting
    Example 2:
    - Memorability score: 0.47
    - Caption: An obscure video of clouds creeping in into the city while cars drive by.
    - Actions: heading, driving

    I would like you to predict the memorability score of the following video using its textual description. Do not output any text, just a number between 0.4 and 1 with a precision of at least two decimal places.
    Take into account that the lowest score is 0.4 and the highest is 1, and most of the videos have a score around 0.8. This is your first example:

    - Caption: {caption}
    - Actions: {actions}
    - Predicted memorability score:
"""}],
    'mem_exp': [
        {"role": "system", "content": "You are an expert in perception-related topics such as memorability. Instructions will be provided to you so you can perform your task."},
        {"role": "user", "content": """
        Hi! I am trying to predict the memorability score of a video using its text description, or caption. Memorability is defined as the ability of a video to be remembered over time (a memorability score close to 1 will represent a video that is likely to be remember whilst lower scores would mean an easily forgettable video). In particular, it is related with the portion of annotators that are able to correctly remember watching the video after some lag in a memory game. All videos are 3 seconds long and are "home videos", meaning that their quality is not very high. This are some examples:

        Example 1:

        - Memorability score: 0.99
        - Caption: A person demonstrates how to fold a dollar to look like a chicken laying an egg.
        - Actions: dropping, folding, falling, crafting
        Example 2:
        - Memorability score: 0.47
        - Caption: An obscure video of clouds creeping in into the city while cars drive by.
        - Actions: heading, driving

        I would like you to provide a brief explanation in which you describe how memorable the video might be. Specify which elements led you to your reasoning and why. This is your first example:

        - Caption: {caption}
        - Actions: {actions}
        - Explanation: 
"""}],
    'grounded_recaption' : [
        {"role": "system", "content": "You are a disciplined automatic content annotator. Instructions will be provided to you so you can perform your task."},
        {"role": "user", "content": """
        Hi! I am trying to predict the memorability score of a video using its text description, or caption. Memorability is defined as the ability of a video to be remembered over time (a memorability score close to 1 will represent a video that is likely to be remember whilst lower scores would mean an easily forgettable video). In particular, it is related with the portion of annotators that are able to correctly remember watching the video after some lag in a memory game. All videos are 3 seconds long and are "home videos", meaning that their quality is not very high. This are some examples:

        Example 1:

        - Memorability score: 0.99
        - Caption: A person demonstrates how to fold a dollar to look like a chicken laying an egg.
        - Actions: dropping, folding, falling, crafting
        Example 2:
        - Memorability score: 0.47
        - Caption: An obscure video of clouds creeping in into the city while cars drive by.
        - Actions: heading, driving

        I would like you to provide an alternative caption to the following video that is more aligned with the memorability task. You must describe the video in a concise and accurate way but using a vocabulary that conveys the how memorable the video is, depending on its score (which is provided). Take into account that the lowest score is 0.4 and the highest is 1, and most videos have scores around 0.8 You must not describe anything that is not mentioned in the original caption. Use a similar number of tokens as the original caption. Here is your original data:

        - Memorability score: {score}
        - Caption: {caption}.
        - Actions: {actions}
        - Alternate caption: 
"""}],
    'grounded_mem_exp' : [
        {"role": "system", "content": "You are an expert in perception-related topics such as memorability. Instructions will be provided to you so you can perform your task."},
        {"role": "user", "content": """
        Hi! I am trying to predict the memorability score of a video using its text description, or caption. Memorability is defined as the ability of a video to be remembered over time (a memorability score close to 1 will represent a video that is likely to be remember whilst lower scores would mean an easily forgettable video). In particular, it is related with the portion of annotators that are able to correctly remember watching the video after some lag in a memory game. All videos are 3 seconds long and are "home videos", meaning that their quality is not very high. This are some examples:

        Example 1:

        - Memorability score: 0.99
        - Caption: A person demonstrates how to fold a dollar to look like a chicken laying an egg.
        - Actions: dropping, folding, falling, crafting
        Example 2:
        - Memorability score: 0.47
        - Caption: An obscure video of clouds creeping in into the city while cars drive by.
        - Actions: heading, driving

        I would like you to provide a brief explanation (1-3 sentences) in which you describe why the video is more or less memorable based on the provided caption, actions and score. Note that the score is given, being 0.4 the lowest for a video, 1 the highest and 0.8 the mode. Specify which elements led you to your reasoning and why. This is your first example:

        - Memorability score: {score}
        - Caption: {caption}
        - Actions: {actions}
        - Explanation: 
"""}]
}